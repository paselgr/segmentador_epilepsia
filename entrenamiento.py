import argparse
import os
import re

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchio as tio
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from modelos import (
    NNUNet3D_Clasico,
    NNUNet3D_Cuantica,
    PlotCallback_Clasico,
    PlotCallback_Cuantica,
    SegmentationModule_Clasico,
    SegmentationModule_Cuantica,
)


class RandomCrop(tio.Transform):
    """
    Función que parchea los escáneres

    Argumentos:
        patch_size (Tuple[int, int, int]): Tamaño del parche que se quiere extraer.
        p_foreground (optional, float): Probabilidad de forzar la extracción de un
            parche que contenga perc_values de los vóxeles de resección.
        perc_values (optional, float): Fracción mínima de vóxeles de la máscara de
            resección que deben de haber en el parche para que se acepte. Se activa
            el p_foreground de los casos.
        n_iter (optional, int): Número máximo de iteraciones en que se ejecutará la
            obtención de un parche que contenga perc_values de los vóxeles de resección
            al activarse p_foreground. En caso de que ninguna sobrepase los perc_values
            vóxeles, se seleccionará la que más tenga.
    """
    def __init__(self, patch_size, p_foreground=0.33, perc_values=0.9, n_iter=5):
        super().__init__()
        self.patch_size = patch_size
        self.p_foreground = p_foreground
        self.perc_values = perc_values
        self.n_iter = n_iter

    def apply_transform(self, subject):
        total_pixel_value = torch.sum(subject.mascara.data.float())#.detach()
        if torch.rand(1) < self.p_foreground:#.detach() < self.p_foreground:
            first_flag = True
            for _ in range(self.n_iter):
                sampler = tio.data.UniformSampler(self.patch_size)
                patch_generator = sampler(subject)
                interm_subject_patch = next(patch_generator)
                mask_pixel_value = torch.sum(interm_subject_patch.mascara.data.float())#.detach()
                if mask_pixel_value >= total_pixel_value * self.perc_values:
                    subject_patch = interm_subject_patch
                    break
                else:
                    if first_flag or mask_pixel_value > max_mask_pixel_value:
                        max_mask_pixel_value = mask_pixel_value
                        subject_patch = interm_subject_patch
                        first_flag = False
        else:
            sampler = tio.data.UniformSampler(self.patch_size)
            patch_generator = sampler(subject)
            subject_patch = next(patch_generator)
        
        return subject_patch


class MyScanDataset(Dataset):
    """
    Dataset que implementa "lazy loading".
    En lugar de rutas, usa objetos Nifti1Image ya cargados.
    """
    def __init__(self, list_of_paths, transformations=None):
        self.paths = list_of_paths
        self.transformations = transformations

    def __len__(self):
        # Devuelve el número total de escáneres referenciados
        return len(self.paths)

    def __getitem__(self, idx):
        # Obtener el objeto Nifti1Image (no la ruta) y la etiqueta
        image_path = self.paths[idx]

        try:
            # Extraer datos y metadatos de Nibabel
            escaner_nib = nib.load(image_path[0])
            mascara_nib = nib.load(image_path[1])

            # .get_fdata() da un array numpy (D, H, W)
            escaner_array = escaner_nib.get_fdata(dtype="float32")
            mascara_array = mascara_nib.get_fdata(dtype="float32")
            # La máscara del cerebro me servirá al final para que no hayan variaciones de intensidad en el fondo del escáner
            mascara_cerebro_array = escaner_array != 0 # Obtengo las posiciones que no son el fondo del escáner
            
            # Convertir a Tensor y añadir un canal (C=1)
            # TorchIO espera (C, D, H, W)
            escaner_tensor = torch.from_numpy(escaner_array).unsqueeze(0)
            mascara_tensor = torch.from_numpy(mascara_array).unsqueeze(0)
            mascara_cerebro_tensor = torch.from_numpy(mascara_cerebro_array).unsqueeze(0)
            
            # Obtener la matriz afín
            escaner_affine_matrix = escaner_nib.affine
            mascara_affine_matrix = mascara_nib.affine
            
            # Creamos el objeto 'ScalarImage' de TorchIO
            escaner_image = tio.ScalarImage(tensor=escaner_tensor, affine=escaner_affine_matrix)
            mascara_image = tio.LabelMap(tensor=mascara_tensor, affine=mascara_affine_matrix)
            mascara_cerebro_image = tio.LabelMap(tensor=mascara_cerebro_tensor, affine=escaner_affine_matrix)
            
            # Juntamos en el objeto Subject (para aplicar todas las transformaciones juntas)
            subject = tio.Subject(
                escaner=escaner_image,
                mascara=mascara_image,
                mascara_cerebro=mascara_cerebro_image
            )
            
        except Exception as e:
            print(f"ERROR: No se pudo procesar la imagen en el índice {idx}.")
            print(f"Detalle: {e}")
            # Si esto falla, el problema es más profundo (ej. datos corruptos)
            return None # Devolvemos None para forzar el error

        # Aplicar Data Augmentation
        if self.transformations:
            subject = self.transformations(subject)

        escaner_datos = subject.escaner.data.float()
        mascara_cerebro_datos = subject.mascara_cerebro.data.float()
        escaner_fondo_sin_ruido_datos = torch.mul(escaner_datos, mascara_cerebro_datos)

        return {
            "pixel_values": escaner_fondo_sin_ruido_datos, 
            "labels":       subject.mascara.data.float()
        }


class VideoDataModule(pl.LightningDataModule):
    def __init__(self, data_root, batch_size=4, num_workers=8):
        """
        Define las transformaciones y parámetros principales.
        """
        super().__init__()
        self.data_root = data_root # Ruta principal
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.patch_size = (112, 131, 98)

        # Transformaciones de entrenamiento
        self.train_transformations = tio.Compose([
            RandomCrop(patch_size=self.patch_size, p_foreground=0.33, perc_values=0.9, n_iter=5),
            tio.RandomAffine(scales=0, degrees=30, p=0.2),
            tio.RandomAffine(scales=(0.85, 1.25), degrees=0, isotropic=True, p=0.2),
            tio.RandomElasticDeformation(num_control_points=7, max_displacement=7.5, p=0.2),
            tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
            tio.RandomNoise(std=(0, 0.1), p=0.15, include=['escaner']),
            tio.RandomBlur(std=(0.5, 1.5), p=0.2, include=['escaner']),
            tio.RandomBiasField(coefficients=0.3, p=0.15, include=['escaner']),
            tio.RandomGamma(log_gamma=(-0.3, 0.4), p=0.1, include=['escaner']),
            tio.RandomAnisotropy(downsampling=(1.5, 2.0), image_interpolation='bspline', p=0.25, include=['escaner'])
        ])

        # Transformaciones de validación
        self.eval_transformations = tio.Compose([
            RandomCrop(patch_size=self.patch_size, p_foreground=0.33, perc_values=0.9, n_iter=5)
        ])
        
        # Listas para guardar las referencias
        self.train_paths = []
        self.train_labels = []
        self.val_paths = []
        self.val_labels = []

    def get_paths_from_folder_numbers(self, folder_numbers, info_list):
        paths = []
        for folder_number in folder_numbers:
            for sorted_file_paths in info_list:
                if f'BraTS-GLI-{folder_number}-' in sorted_file_paths[0]:
                    paths.append(sorted_file_paths)
        return paths

    def setup(self, stage=None): # Se ejecuta una única vez al ejecutar la función
        """
        Busca las referencias (paths) y crea los Datasets.
        Esto se ejecuta una vez en cada 'fit' o 'test'.
        """
        if stage == 'fit' or stage is None:
            info_list = []
            folder_numbers = []
            for folder in os.listdir(self.data_root):
                folder_number = re.search(r'(?<=BraTS-GLI-)\d+(?=-\d+)', folder).group(0)
                if folder_number not in folder_numbers:
                    folder_numbers.append(folder_number)
            for folder in os.listdir(self.data_root):
                folder_path = os.path.join(self.data_root, folder)
                files = os.listdir(folder_path)
                if len(files) < 2:
                    print(f'En la ruta {folder_path} faltan archivos:\n{files=}')
                else:
                    file_paths = [os.path.join(folder_path, file) for file in files]
                    # Ordenamos los archivos: 1.Escáner, 2.Máscara
                    sorted_file_paths = sorted(file_paths, key=lambda x: x.endswith('-t1n.nii.gz'), reverse=True)
                    info_list.append(sorted_file_paths)

            train_folder_numbers, intermediate_folder_numbers = train_test_split(
                folder_numbers,
                test_size=0.3,
            )

            val_folder_numbers, test_folder_numbers = train_test_split(
                intermediate_folder_numbers,
                test_size=0.5,
            )

            train_paths = self.get_paths_from_folder_numbers(train_folder_numbers, info_list)
            val_paths   = self.get_paths_from_folder_numbers(val_folder_numbers,   info_list)
            test_paths  = self.get_paths_from_folder_numbers(test_folder_numbers,  info_list)

            # Instanciar los datasets con las rutas y transformaciones
            self.train_dataset = MyScanDataset(
                list_of_paths=train_paths,
                transformations=self.train_transformations,
            )
            
            self.val_dataset = MyScanDataset(
                list_of_paths=val_paths,
                transformations=self.eval_transformations,
            )

            self.test_dataset = MyScanDataset(
                list_of_paths=test_paths,
                transformations=None,
            )

    def train_dataloader(self):
        """
        Crea el DataLoader de entrenamiento.
        """
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,                 # Mezcla los datos en cada época
            num_workers=self.num_workers, # Carga datos en paralelo
            pin_memory=True               # Acelera transferencia a GPU
        )

    def val_dataloader(self):
        """
        Crea el DataLoader de validación.
        """
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, # No se mezcla en validación
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        """
        Crea el DataLoader de test.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=1, # batch_size = 1 porque calculamos el test real de imágenes con distintos tamaños (si agrupamos da error)
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

def main():
    parser = argparse.ArgumentParser(description='Entrenamiento con el dataset BraTS.')
    parser.add_argument('--model_type', type=str, choices=['Clasico', 'Cuantica'], required=True,
                        help='Especifica qué tipo de modelo se entrena (Clasico o Cuantica)')
    parser.add_argument('-e', '--max_epochs', default=1000, type=int, required=False,
                        help='Número de épocas del entrenamiento. Por defecto 1000.')
    parser.add_argument('-b', '--batch_size', default=8, type=int, required=False,
                        help='Número de elementos por lote. Por defecto 8.')
    parser.add_argument('-w', '--num_workers', default=20, type=int, required=False,
                        help='Número de workers. Por defecto 20.')
    args = parser.parse_args()

    if args.model_type == 'Clasico':
        model = NNUNet3D_Clasico(input_channels=1, num_classes=1)
        pl_model = SegmentationModule_Clasico(model, learning_rate=1e-2, max_epochs=args.max_epochs)
        plot_callback = PlotCallback_Clasico()
    if args.model_type == 'Cuantica':
        model = NNUNet3D_Cuantica(input_channels=1, num_classes=1, number_quantum_layers=5)
        pl_model = SegmentationModule_Cuantica(model, learning_rate=1e-2, max_epochs=args.max_epochs)
        plot_callback = PlotCallback_Cuantica()


    # Mover el modelo a GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    early_stopping_callback = EarlyStopping('val_dice', min_delta=0.0, patience=300, mode='max')

    csv_logger = CSVLogger(save_dir='')

    filename_base = f"best_model-{model._get_name()}-n_epochs={args.max_epochs:02d}-"

    # Instanciar el DataModule
    data_module = VideoDataModule(
        data_root='datos_preprocesados_BraTS',
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_dice", # Métrica a monitorizar
        mode="max",
        filename=filename_base+"{epoch:02d}-{val_dice:.2f}"
    )

    # Entrenar
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        callbacks=[checkpoint_callback,
                early_stopping_callback,
                plot_callback],
        logger=csv_logger,
        
        # Optimizaciones
        precision="16-mixed", # Reduce memoria a la mitad (aprox)
        accumulate_grad_batches=2, # Simula Batch Size: 2 * batch_size. Por defecto 2 * 8 = 16

        benchmark=True
    )
    trainer.fit(pl_model, data_module)


    # Ejecutar las pruebas
    print("--- Entrenamiento finalizado ---)")
    print("--- Ejecutando pruebas con el MEJOR checkpoint... ---")

    results = trainer.test(model=pl_model, datamodule=data_module, ckpt_path="best", weights_only=False)

    print("Resultados de la prueba:")
    print(results)

    # Guardamos los resultados
    metrics = pd.read_csv(os.path.join(trainer.logger.log_dir, 'metrics.csv'))

    test_dice_avg  = [test_dice_avg for test_dice_avg in metrics['test_dice_avg'] if not np.isnan(test_dice_avg)][0]
    test_dice_std  = [test_dice_std for test_dice_std in metrics['test_dice_std'] if not np.isnan(test_dice_std)][0]
    test_dice_med  = [test_dice_med for test_dice_med in metrics['test_dice_med'] if not np.isnan(test_dice_med)][0]
    test_dice_iqr  = [test_dice_iqr for test_dice_iqr in metrics['test_dice_iqr'] if not np.isnan(test_dice_iqr)][0]
    test_loss  = [test_loss for test_loss in metrics['test_loss'] if not np.isnan(test_loss)][0]

    version = trainer.logger.version

    # Guardamos los datos del entrenamiento
    os.makedirs(os.path.join('resultados', f'version_{version}'), exist_ok=True)
    ruta_resultados = os.path.join('resultados', f'version_{version}', 'resultados.csv')
    os.makedirs(os.path.dirname(ruta_resultados), exist_ok=True)

    # Creamos el csv
    datos_dict = {
        'Model_Type': [args.model_type],
        'Execution': ['Training'],
        'Version': [version],
        'Test_Dice_Avg': [test_dice_avg],
        'Test_Dice_Std': [test_dice_std],
        'Test_Dice_Med': [test_dice_med],
        'Test_Dice_IQR': [test_dice_iqr],
        'Test_Loss': [test_loss]
    }
    df = pd.DataFrame(datos_dict)
    df.to_csv(ruta_resultados, index=False, encoding='utf-8')

if __name__ == '__main__':
    main()
