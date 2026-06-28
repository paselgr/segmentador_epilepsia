import argparse
import os
import random
import re
import shutil

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchio as tio
from lightning.pytorch.callbacks import DeviceStatsMonitor, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset

from modelos import (
    NNUNet3D_Clasico,
    NNUNet3D_Cuantica,
    PlotCallback_Clasico,
    PlotCallback_Cuantica,
    SegmentationModule_Clasico,
    SegmentationModule_Cuantica,
)


def crear_aumento_offline(lista_rutas_entrada, cantidad, carpeta_salida, train_transformations=None):
    """
    Crea una carpeta con el aumento de datos de los archivos
    seleccionados para el aumento de datos offline.
    Si la carpeta ya existe (de un fine-tuning anterior) la
    elimina.
    
    Argumentos:
        *lista_rutas_entrada (str): Conjunto de las listas de las cuales
            se realizará el aumento de datos.
        cantidad (int): Número de veces que se aplicará el aumento de datos,
            p.ej. cantidad=2 indica que cada archivo será aumentado tres veces
            distintas.
        carpeta_salida (str): Nombre de la carpeta donde se guardará el
            aumento de datos.
        train_transformations (optional, torchio.Compose | None): Transformaciones
            que se utilizarán en el aumento de datos. Si se indica None se utilizarán
            las transformaciones por defecto.
    """
    # Eliminamos la carpeta con el mismo nombre si ya existía
    if os.path.exists(carpeta_salida):
        shutil.rmtree(carpeta_salida)
    os.mkdir(carpeta_salida)
    if train_transformations == None:
        train_transformations = tio.Compose([
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
    
    # Repetimos la lista entera tantas veces como indique el parámetro cantidad
    for _ in range(cantidad):
        # El sufijo servirá para diferenciar la iteración a la que pertenecen los datos
        n_sufijo_subcarpeta = int(len(os.listdir(carpeta_salida))/len(lista_rutas_entrada))
        for rutas in lista_rutas_entrada:
            try:
                escaner_nib = nib.load(rutas[0])
                mascara_nib = nib.load(rutas[1])

                escaner_array = escaner_nib.get_fdata(dtype="float32")
                mascara_array = mascara_nib.get_fdata(dtype="float32")
                # La máscara del cerebro me servirá al final para que no hayan variaciones de intensidad en el fondo del escáner
                mascara_cerebro_array = escaner_array != 0 # Obtengo las posiciones que no son el fondo del escáner
                
                escaner_tensor = torch.from_numpy(escaner_array).unsqueeze(0)
                mascara_tensor = torch.from_numpy(mascara_array).unsqueeze(0)
                mascara_cerebro_tensor = torch.from_numpy(mascara_cerebro_array).unsqueeze(0)
                
                escaner_affine_matrix = escaner_nib.affine
                mascara_affine_matrix = mascara_nib.affine

                escaner_header = escaner_nib.header
                mascara_header = mascara_nib.header
                
                escaner_image = tio.ScalarImage(tensor=escaner_tensor, affine=escaner_affine_matrix)
                mascara_image = tio.LabelMap(tensor=mascara_tensor, affine=mascara_affine_matrix)
                mascara_cerebro_image = tio.LabelMap(tensor=mascara_cerebro_tensor, affine=escaner_affine_matrix)
                
                subject = tio.Subject(
                    escaner=escaner_image,
                    mascara=mascara_image,
                    mascara_cerebro=mascara_cerebro_image
                )
            except:
                print('Error en la carga de datos del aumento de datos offline')

            subject = train_transformations(subject)

            escaner_datos = subject.escaner.data.float()
            mascara_cerebro_datos = subject.mascara_cerebro.data.float()
            escaner_fondo_sin_ruido_datos = torch.mul(escaner_datos, mascara_cerebro_datos)

            escaner_mod_affine_matrix = subject.escaner.affine
            mascara_mod_affine_matrix = subject.mascara.affine

            escaner_mod_data_array = np.array(escaner_fondo_sin_ruido_datos.squeeze())
            mascara_mod_data_array = np.array(subject.mascara.data.float().squeeze())

            escaner_mod_nib = nib.Nifti1Image(escaner_mod_data_array, affine=escaner_mod_affine_matrix, header=escaner_header)
            mascara_mod_nib = nib.Nifti1Image(mascara_mod_data_array, affine=mascara_mod_affine_matrix, header=mascara_header)

            # El prefijo sirve para no sobrescribir las carpetas si cantidad > 1
            subcarpeta_salida = f'{rutas[0].split("/")[1]}_{n_sufijo_subcarpeta}'
            ruta_subcarpeta_salida = os.path.join(carpeta_salida, subcarpeta_salida)
            assert not os.path.isdir(ruta_subcarpeta_salida), f'La subcarpeta {subcarpeta_salida} ya existe.'
            os.mkdir(ruta_subcarpeta_salida)

            ruta_salida_escaner = os.path.join(ruta_subcarpeta_salida, rutas[0].split('/')[2])
            ruta_salida_mascara = os.path.join(ruta_subcarpeta_salida, rutas[1].split('/')[2])

            nib.save(escaner_mod_nib, ruta_salida_escaner)
            nib.save(mascara_mod_nib, ruta_salida_mascara)
    # Devolver las rutas ordenadas
    data_off_paths = []
    for folder in os.listdir(carpeta_salida):
        lista = []
        for file in os.listdir(os.path.join(carpeta_salida, folder)):
            lista.append(os.path.join(carpeta_salida, folder, file))
            lista = sorted(lista, key=lambda x: x.endswith('-t1n.nii.gz'), reverse=True)
        data_off_paths.append(lista)
    return data_off_paths


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
    def __init__(self, info_list, batch_size=4, num_workers=8):
        """
        Define las transformaciones y parámetros principales.
        """
        super().__init__()
        self.info_list = info_list # Lista de rutas
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.patch_size = (112, 131, 98)
        self.stratify = True

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

    def setup(self, stage=None): # Se ejecuta una única vez al ejecutar la función
        """
        Busca las referencias (paths) y crea los Datasets.
        Esto se ejecuta una vez en cada 'fit' o 'test'.
        """

        if stage == 'fit' or stage is None:
            train_paths, val_paths, test_paths = self.info_list

            # Creamos el aumento de datos offline
            train_off_paths = crear_aumento_offline(train_paths, 1, 'datos_aum_off_EPISURG_posop_entrenamiento')
            val_off_paths   = crear_aumento_offline(val_paths,   2, 'datos_aum_off_EPISURG_posop_validacion')
            
            train_paths = train_paths + train_off_paths
            val_paths   = val_paths   + val_off_paths

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


def mezclador_con_semilla(lista, semilla=False):
    """
    Creamos un mezclador local con su propia semilla.
    
    Argumentos:
        *lista: Conjunto de datos que se mezclarán.
        semilla (optional, int | False): Si el dato proporcionado es un entero
            indica la semilla de la función de mezclado para facilitar
            las comparaciones, si es False no se utiliza semilla.
    """
    if semilla is False:
        rng = random.Random()
    else:
        rng = random.Random(semilla)
    rng.shuffle(lista)
    return lista


def obtener_conjuntos(data_root, n_conjuntos, semilla=False):
    """"
    Estratifica los datos en n conjuntos distintos.

    Argumentos:
        data_root (str): Carpeta base donde se encuentran los archivos
             a estratificar, se espera que tenga unas características determinadas
             que cumple el conjunto preprocesado de EPISURG.
        n (int): Número de conjuntos en los que se estratificarán los datos.
        semilla (optional, int | False): Si es entero semilla de la mezcla de los
            datos tras la estratificación, si es False no se utiliza semilla.
    """
    def indicar_datos_tipo(sorted_file_paths):
        subject_ID = sorted_file_paths[0].split('/')[1].split('_')[0]
        subject_ID = re.search(r'sub-(\d){4}', sorted_file_paths[0]).group(0)
        episurg_data_df = pd.read_csv(os.path.join('EPISURG', 'subjects.csv'))

        episurg_type = str(episurg_data_df[episurg_data_df['Subject'] == subject_ID]['Type']).split('\n')[0].lstrip(r'\d ')
        episurg_type = re.search(r'(?![\d ])[\w ]*(?=\nName)', str(episurg_data_df[episurg_data_df['Subject'] == subject_ID]['Type'])).group(0)

        sorted_file_paths.append(episurg_type)
        return sorted_file_paths
    
    def n_estratificacion(elementos, clases, n):
        strats_info = np.unique(clases, return_counts=True)

        conjuntos = [[] for _ in range(n)]
        conjunto_intermedio = []
        for clase, n_clase in zip(*strats_info):
            i = 0
            for elem in elementos:
                if elem[2] == clase:
                    if i < n_clase//n*n:
                        conjuntos[i%n].append(elem)
                        i += 1
                    else:
                        conjunto_intermedio.append(elem)
        for i in range(len(conjunto_intermedio)):
            conjuntos[i%n].append(conjunto_intermedio[i])
        return conjuntos
    
    info_list = []
    for folder in os.listdir(data_root):
        folder_path = os.path.join(data_root, folder)
        files = os.listdir(folder_path)
        assert len(files) >= 2, f'En la ruta {folder_path} faltan archivos:\n{files=}'
        file_paths = [os.path.join(folder_path, file) for file in files]
        # Ordenamos los archivos: 1.Escáner, 2.Máscara
        sorted_file_paths = sorted(file_paths, key=lambda x: x.endswith('-t1mri-1.nii.gz'), reverse=True)
        sorted_file_paths = indicar_datos_tipo(sorted_file_paths) # Estratificamos
        info_list.append(sorted_file_paths)
    
    if semilla is False:
        info_list = mezclador_con_semilla(info_list)
    else:
        info_list = mezclador_con_semilla(info_list, False)
    clases = [elem_list[2] for elem_list in info_list]

    conjuntos = n_estratificacion(info_list, clases, n_conjuntos)
    return conjuntos

parser = argparse.ArgumentParser(description='Entrenamiento con el dataset BraTS.')
parser.add_argument('--model_type', type=str, choices=['Clasico', 'Cuantica'], required=True,
                    help='Especifica qué tipo de modelo se realiza el fine-tuning (Clasico o Cuantica)')
parser.add_argument('-v', '--version', type=int, required=True,
                    help='Versión del modelo sobre el cual se realizará fine-tuning')
parser.add_argument('-n', '--num_folds', default=6, type=int, required=False,
                    help='Número conjuntos para la validación cruzada. Por defecto 6.')
parser.add_argument('-s', '--seed', default=False, type=int, required=False,
                    help='Semilla para comparar modelos. Por defecto no se activa.')
parser.add_argument('-e', '--max_epochs', default=1000, type=int, required=False,
                    help='Número de épocas del entrenamiento. Por defecto 1000.')
parser.add_argument('-b', '--batch_size', default=8, type=int, required=False,
                    help='Número de elementos por lote. Por defecto 8.')
parser.add_argument('-w', '--num_workers', default=20, type=int, required=False,
                    help='Número de workers. Por defecto 20.')
args = parser.parse_args()

conjuntos = obtener_conjuntos(data_root='datos_preprocesados_EPISURG', n_conjuntos=args.num_folds, semilla=args.seed) ### argparse
for i in range(args.num_folds):
    conjuntos[i] = mezclador_con_semilla(conjuntos[i], args.seed) ### argparse

lista_versiones = []
test_files = []

for i in range(args.num_folds):
    test_paths = conjuntos[i]
    val_paths  = conjuntos[(i+1)%args.num_folds]
    train_paths = []
    for j in range(args.num_folds):
        if j not in [i, (i+1)%args.num_folds]:
            train_paths += conjuntos[j]
    train_paths = mezclador_con_semilla(train_paths, args.seed)
    val_paths   = mezclador_con_semilla(val_paths,   args.seed)
    test_paths  = mezclador_con_semilla(test_paths,  args.seed)
    info_list = train_paths, val_paths, test_paths
    test_files.append(test_paths)

    if args.model_type == 'Clasico':
        model = NNUNet3D_Clasico(input_channels=1, num_classes=1)
        pl_model = SegmentationModule_Clasico(model, learning_rate=1e-3, max_epochs=args.max_epochs)
        plot_callback = PlotCallback_Clasico()
    if args.model_type == 'Cuantica':
        model = NNUNet3D_Cuantica(input_channels=1, num_classes=1, number_quantum_layers=5)
        pl_model = SegmentationModule_Cuantica(model, learning_rate=1e-3, max_epochs=args.max_epochs)
        plot_callback = PlotCallback_Cuantica()

    # Cargar el modelo y moverlo a GPU
    version = args.version
    n_checkpoint_files = 0
    for root, folders, files in os.walk(os.path.join('lightning_logs', f'version_{version}')):
        for file in files:
            if file.endswith('.ckpt'):
                checkpoint_path = os.path.join(root, file)
                n_checkpoint_files += 1
                assert n_checkpoint_files == 1, f'Hay más de un archivo de pesos en la carpeta {root}'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
    new_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        new_key = key.replace('model.', '', 1) if key.startswith('model.') else key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    early_stopping_callback = EarlyStopping('val_dice', min_delta=0.0, patience=300, mode='max')

    csv_logger = CSVLogger(save_dir='')

    max_epochs = 1000 ### argparse
    filename_base = f"best_model-{model._get_name()}-n_epochs={max_epochs:02d}-"

    data_module = VideoDataModule(
        info_list=info_list,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_dice", # Métrica a monitorizar
        mode="max",
        save_top_k=5, # Guardamos las 5 mejores épocas
        filename=filename_base+"{epoch:02d}" # Con el nombre completo da problema para acceder a los checkpoints
    )


    # 3. Entrenas
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        callbacks=[checkpoint_callback,
                early_stopping_callback,
                plot_callback,
                DeviceStatsMonitor()],
        logger=csv_logger,

        # Optimizaciones
        precision="16-mixed",       # Reduce memoria a la mitad (aprox)
        accumulate_grad_batches=2,  # Simula Batch Size = 8 * 2 = 16
        
        # gradient clipping
        gradient_clip_val=1.0,          # Valor máximo permitido.
        gradient_clip_algorithm="norm", # "norm" recorta basándose en la norma global de los gradientes. "value" corta por valor absoluto.

        benchmark=True
    )
    trainer.fit(pl_model, data_module)


    # Ejecutar las pruebas
    print("--- Entrenamiento finalizado ---")
    print("--- Calculando ensemble de los 5 mejores checkpoints... ---")

    # Leemos los archivos reales del disco
    ckpt_dir = checkpoint_callback.dirpath
    rutas_checkpoints = [
        os.path.join(ckpt_dir, f) 
        for f in os.listdir(ckpt_dir) 
        if f.endswith('.ckpt')
    ]

    # Inicializar un nuevo state_dict vacío
    ensemble_state_dict = None
    num_checkpoints = len(rutas_checkpoints)

    for ruta in rutas_checkpoints:
        # Cargar el checkpoint
        ckpt = torch.load(ruta, map_location=torch.device('cpu'), weights_only=False)
        ckpt_state_dict = ckpt['state_dict']
        
        # Si es el primero, clonamos la estructura dividiendo por el número total
        if ensemble_state_dict is None:
            ensemble_state_dict = {
                k: v.clone().float() / num_checkpoints 
                for k, v in ckpt_state_dict.items()
            }
        else:
            # Sumar la fracción correspondiente de los pesos de los siguientes checkpoints
            for k, v in ckpt_state_dict.items():
                ensemble_state_dict[k] += v.float() / num_checkpoints

    # Cargar los pesos promediados en el modelo Lightning
    pl_model.load_state_dict(ensemble_state_dict)

    print("--- Pesos promediados cargados. Ejecutando pruebas... ---")
    # Realizar el test usando el modelo con los pesos del ensemble
    # Omitimos ckpt_path="best" para forzar el uso de los pesos que acabamos de cargar en memoria
    results = trainer.test(model=pl_model, datamodule=data_module)

    print("Resultados de la prueba (Weight Averaging):")
    print(results)

    # Guardamos los resultados
    metrics = pd.read_csv(os.path.join(trainer.logger.log_dir, 'metrics.csv'))

    test_dice_avg  = [test_dice_avg for test_dice_avg in metrics['test_dice_avg'] if not np.isnan(test_dice_avg)][-1]
    test_dice_std  = [test_dice_std for test_dice_std in metrics['test_dice_std'] if not np.isnan(test_dice_std)][-1]
    test_dice_med  = [test_dice_med for test_dice_med in metrics['test_dice_med'] if not np.isnan(test_dice_med)][-1]
    test_dice_iqr  = [test_dice_iqr for test_dice_iqr in metrics['test_dice_iqr'] if not np.isnan(test_dice_iqr)][-1]
    test_loss  = [test_loss for test_loss in metrics['test_loss'] if not np.isnan(test_loss)][-1]

    version = trainer.logger.version

    # Guardamos los datos del fine-tuning
    os.makedirs(os.path.join('resultados', f'version_{version}'), exist_ok=True)
    ruta_resultados = os.path.join('resultados', f'version_{version}', 'resultados.csv')
    os.makedirs(os.path.dirname(ruta_resultados), exist_ok=True)

    # Creamos el csv
    datos_dict = {
        'Model_Type': [args.model_type],
        'Execution': 'Fine-tuning',
        'Version': [version],
        'Test_Dice_Avg': [test_dice_avg],
        'Test_Dice_Std': [test_dice_std],
        'Test_Dice_Med': [test_dice_med],
        'Test_Dice_IQR': [test_dice_iqr],
        'Test_Loss': [test_loss]
    }
    df = pd.DataFrame(datos_dict)
    df.to_csv(ruta_resultados, index=False, encoding='utf-8')

    lista_versiones.append(version)
versiones = '-'.join([str(version) for version in lista_versiones])
ruta_carpeta_multiples_graficas = os.path.join('graficas', f'version_{versiones}')
os.makedirs(ruta_carpeta_multiples_graficas, exist_ok=True)

total_test_dice = []
for version in lista_versiones:
    with open(os.path.join('resultados_enteros', f'version_{version}', 'resultados_enteros.txt'), mode='r') as f:
        next(f)
        for line in f:
            total_test_dice.append(line.strip())
total_test_dice = [float(num) for num in total_test_dice]

# Guardamos el Boxplt de todos los datos
os.makedirs(os.path.join('graficas', f'version_{versiones}'), exist_ok=True)
ruta_boxplot = os.path.join('graficas', f'version_{versiones}', 'test_Dice_boxplot.png')
os.makedirs(os.path.dirname(ruta_boxplot), exist_ok=True)

plt.figure()
plt.boxplot(total_test_dice, positions=[1])
plt.title('All test Dice coefficients')
plt.xticks([1], ['']) # Ocultar las etiquetas del eje X
plt.grid(True)
plt.savefig(ruta_boxplot)
plt.close()

# Guardamos los datos globales del fine-tuning
avg_total_dice = np.mean(total_test_dice)
std_total_dice = np.std(total_test_dice)
med_total_dice = np.median(total_test_dice)
iqr_total_dice = np.quantile(total_test_dice, 0.75) - np.quantile(total_test_dice, 0.25)

os.makedirs(os.path.join('resultados', f'version_{versiones}'), exist_ok=True)
ruta_resultados = os.path.join('resultados', f'version_{versiones}', 'resultados_globales.csv')
os.makedirs(os.path.dirname(ruta_resultados), exist_ok=True)

# Creamos el csv
datos_dict = {
    'Model_Type': [args.model_type],
    'Execution': ['Fine_tuning'],
    'Version': [versiones],
    'Test_Dice_Avg': [avg_total_dice],
    'Test_Dice_Std': [std_total_dice],
    'Test_Dice_Med': [med_total_dice],
    'Test_Dice_IQR': [iqr_total_dice]
}
df = pd.DataFrame(datos_dict)
df.to_csv(ruta_resultados, index=False, encoding='utf-8')

# Guardamos los coeficientes de Dice individuales
os.makedirs(os.path.join('resultados_enteros', f'version_{versiones}'), exist_ok=True)
ruta_resultados_enteros = os.path.join('resultados_enteros', f'version_{versiones}', f'fine-tuning_{args.model_type}_{versiones}.csv') ### argparse
os.makedirs(os.path.dirname(ruta_resultados_enteros), exist_ok=True)

rutas_test = []
iteraciones_pertenece = []
iteracion_pertenece = 1
for rutas_archivos_iteracion in test_files:
    for tupla_rutas in rutas_archivos_iteracion:
        for ruta_archivo in tupla_rutas:
            if ruta_archivo.endswith('-t1mri-1.nii.gz'):
                rutas_test.append(ruta_archivo.split('/')[-1])
                iteraciones_pertenece.append(iteracion_pertenece)
    iteracion_pertenece += 1

# Creamos el csv
datos_dict = {
    'Archivo': rutas_test,
    'Coeficiente_de_Dice': total_test_dice,
    'Iteracion': iteraciones_pertenece
}
df = pd.DataFrame(datos_dict)
df.to_csv(ruta_resultados_enteros, index=False, encoding='utf-8')
