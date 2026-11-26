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

from models import (
    NNUNet3D_Classic,
    NNUNet3D_Hybrid,
    PlotCallback_Classic,
    PlotCallback_Hybrid,
    SegmentationModule_Classic,
    SegmentationModule_Hybrid,
)


def crear_aumento_offline(lista_rutas_entrada, cantidad, carpeta_salida, train_transformations=None):
    """
    Creates a folder with data augmentation of the selected files
    for offline data augmentation.
    If the folder already exists (from a previous fine-tuning run),
    it removes it.
    
    Args:
        *lista_rutas_entrada (str): Collection of lists from which
            data augmentation will be performed.
        cantidad (int): Number of times data augmentation will be applied,
            e.g., cantidad=2 indicates each file will be augmented two
            different times.
        carpeta_salida (str): Name of the folder where the augmented data
            will be saved.
        train_transformations (optional, torchio.Compose | None): Transformations
            to use for data augmentation. If None, default transformations
            will be used.
    """
    # Remove the folder with the same name if it already exists
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
    
    # Repeat the entire list as many times as indicated by the cantidad parameter
    for _ in range(cantidad):
        # The suffix distinguishes which iteration the data belongs to
        n_sufijo_subcarpeta = int(len(os.listdir(carpeta_salida))/len(lista_rutas_entrada))
        for rutas in lista_rutas_entrada:
            try:
                escaner_nib = nib.load(rutas[0])
                mascara_nib = nib.load(rutas[1])

                escaner_array = escaner_nib.get_fdata(dtype="float32")
                mascara_array = mascara_nib.get_fdata(dtype="float32")
                # The brain mask will be used at the end so there are no intensity variations in the scan background
                mascara_cerebro_array = escaner_array != 0 # Get positions that are not the scan background
                
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
                print('Error loading offline data augmentation data')

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

            # The prefix avoids overwriting folders when cantidad > 1
            subcarpeta_salida = f'{rutas[0].split("/")[1]}_{n_sufijo_subcarpeta}'
            ruta_subcarpeta_salida = os.path.join(carpeta_salida, subcarpeta_salida)
            assert not os.path.isdir(ruta_subcarpeta_salida), f'Subfolder {subcarpeta_salida} already exists.'
            os.mkdir(ruta_subcarpeta_salida)

            ruta_salida_escaner = os.path.join(ruta_subcarpeta_salida, rutas[0].split('/')[2])
            ruta_salida_mascara = os.path.join(ruta_subcarpeta_salida, rutas[1].split('/')[2])

            nib.save(escaner_mod_nib, ruta_salida_escaner)
            nib.save(mascara_mod_nib, ruta_salida_mascara)
    # Return the sorted paths
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
    Function that patches the scans.

    Args:
        patch_size (Tuple[int, int, int]): Size of the patch to extract.
        p_foreground (optional, float): Probability of forcing the extraction of a
            patch containing perc_values of the resection voxels.
        perc_values (optional, float): Minimum fraction of voxels from the resection
            mask that must be present in the patch for it to be accepted. Triggered in
            p_foreground of cases.
        n_iter (optional, int): Maximum number of iterations to run while trying to
            obtain a patch containing perc_values of resection voxels when p_foreground
            is triggered. If none exceed perc_values voxels, the one with the highest
            count is selected.
    """
    def __init__(self, patch_size, p_foreground=0.33, perc_values=0.9, n_iter=5):
        super().__init__()
        self.patch_size = patch_size
        self.p_foreground = p_foreground
        self.perc_values = perc_values
        self.n_iter = n_iter

    def apply_transform(self, subject):
        total_pixel_value = torch.sum(subject.mascara.data.float())
        if torch.rand(1) < self.p_foreground:
            first_flag = True
            for _ in range(self.n_iter):
                sampler = tio.data.UniformSampler(self.patch_size)
                patch_generator = sampler(subject)
                interm_subject_patch = next(patch_generator)
                mask_pixel_value = torch.sum(interm_subject_patch.mascara.data.float())
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
    Dataset implementing "lazy loading".
    Instead of paths, uses already-loaded Nifti1Image objects.
    """
    def __init__(self, list_of_paths, transformations=None):
        self.paths = list_of_paths
        self.transformations = transformations

    def __len__(self):
        # Returns the total number of referenced scans
        return len(self.paths)

    def __getitem__(self, idx):
        # Gets the Nifti1Image object (not the path) and the label
        image_path = self.paths[idx]

        try:
            # Extract data and metadata from Nibabel
            escaner_nib = nib.load(image_path[0])
            mascara_nib = nib.load(image_path[1])

            # .get_fdata() returns a numpy array (D, H, W)
            escaner_array = escaner_nib.get_fdata(dtype="float32")
            mascara_array = mascara_nib.get_fdata(dtype="float32")
            # The brain mask will be used at the end so there are no intensity variations in the scan background
            mascara_cerebro_array = escaner_array != 0 # Gets positions that are not the scan background
            
            # Convert to Tensor and add a channel (C=1)
            # TorchIO expects (C, D, H, W)
            escaner_tensor = torch.from_numpy(escaner_array).unsqueeze(0)
            mascara_tensor = torch.from_numpy(mascara_array).unsqueeze(0)
            mascara_cerebro_tensor = torch.from_numpy(mascara_cerebro_array).unsqueeze(0)
            
            # Get the affine matrix
            escaner_affine_matrix = escaner_nib.affine
            mascara_affine_matrix = mascara_nib.affine
            
            # Create the TorchIO 'ScalarImage' object
            escaner_image = tio.ScalarImage(tensor=escaner_tensor, affine=escaner_affine_matrix)
            mascara_image = tio.LabelMap(tensor=mascara_tensor, affine=mascara_affine_matrix)
            mascara_cerebro_image = tio.LabelMap(tensor=mascara_cerebro_tensor, affine=escaner_affine_matrix)
            
            # Combine into a Subject object (to apply all transformations together)
            subject = tio.Subject(
                escaner=escaner_image,
                mascara=mascara_image,
                mascara_cerebro=mascara_cerebro_image
            )
            
        except Exception as e:
            print(f"ERROR: Could not process the image at index {idx}.")
            print(f"Details: {e}")
            # If this fails, the issue is deeper (e.g., corrupted data)
            return None # Return None to force the error

        # Apply Data Augmentation
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
        Defines the main transformations and parameters.
        """
        super().__init__()
        self.info_list = info_list # List of paths
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.patch_size = (112, 131, 98)
        self.stratify = True

        # Training transformations
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

        # Validation transformations
        self.eval_transformations = tio.Compose([
            RandomCrop(patch_size=self.patch_size, p_foreground=0.33, perc_values=0.9, n_iter=5)
        ])

    def setup(self, stage=None): # Runs only once when the function is called
        """
        Finds references (paths) and creates the Datasets.
        This runs once per 'fit' or 'test'.
        """

        if stage == 'fit' or stage is None:
            train_paths, val_paths, test_paths = self.info_list

            # Create offline data augmentation
            train_off_paths = crear_aumento_offline(train_paths, 1, 'datos_aum_off_EPISURG_posop_entrenamiento')
            val_off_paths   = crear_aumento_offline(val_paths,   2, 'datos_aum_off_EPISURG_posop_validacion')
            
            train_paths = train_paths + train_off_paths
            val_paths   = val_paths   + val_off_paths

            # Instantiate datasets with paths and transformations
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
        Creates the training DataLoader.
        """
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,                 # Shuffles data every epoch
            num_workers=self.num_workers, # Loads data in parallel
            pin_memory=True               # Speeds up transfer to GPU
        )

    def val_dataloader(self):
        """
        Create the validation DataLoader.
        """
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,               # Do not shuffle during validation
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        """
        Create the test DataLoader.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=1, # batch_size = 1 because we evaluate the real test on images of different sizes (batching causes an error)
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


def mezclador_con_semilla(lista, semilla=False):
    """
    Create a local shuffler with its own seed.
    
    Args:
        *lista: Dataset to be shuffled.
        semilla (optional, int | False): If an integer is provided,
            specifies the seed for the shuffle function to make
            comparisons easier; if False, no seed is used.
    """
    if semilla is False:
        rng = random.Random()
    else:
        rng = random.Random(semilla)
    rng.shuffle(lista)
    return lista


def obtener_conjuntos(data_root, n_conjuntos, semilla=False):
    """"
    Stratify the data into n distinct sets.

    Args:
        data_root (str): Base folder containing the files to stratify,
            expected to follow certain characteristics that satisfies
            the preprocessed EPISURG dataset.
        n (int): Number of sets into which the data will be stratified.
        semilla (optional, int | False): If an integer, seed for shuffling
            the data after stratification; if False, no seed is used.
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
        assert len(files) >= 2, f'In the path {folder_path} there are missing files:\n{files=}'
        file_paths = [os.path.join(folder_path, file) for file in files]
        # Sort files: 1.Scan, 2.Mask
        sorted_file_paths = sorted(file_paths, key=lambda x: x.endswith('-t1mri-1.nii.gz'), reverse=True)
        sorted_file_paths = indicar_datos_tipo(sorted_file_paths) # Stratify
        info_list.append(sorted_file_paths)
    
    if semilla is False:
        info_list = mezclador_con_semilla(info_list)
    else:
        info_list = mezclador_con_semilla(info_list, False)
    clases = [elem_list[2] for elem_list in info_list]

    conjuntos = n_estratificacion(info_list, clases, n_conjuntos)
    return conjuntos

parser = argparse.ArgumentParser(description='Training with the EPISURG dataset.')
parser.add_argument('--model_type', type=str, choices=['Classic', 'Hybrid'], required=True,
                    help='Specify which model type to fine-tune (Classic or Hybrid)')
parser.add_argument('-v', '--version', type=int, required=True,
                    help='Model version to fine-tune')
parser.add_argument('-n', '--num_folds', default=6, type=int, required=False,
                    help='Number of sets for cross-validation. Default is 6.')
parser.add_argument('-s', '--seed', default=False, type=int, required=False,
                    help='Seed for comparing models. Disabled by default.')
parser.add_argument('-e', '--max_epochs', default=1000, type=int, required=False,
                    help='Number of training epochs. Default is 1000.')
parser.add_argument('-b', '--batch_size', default=8, type=int, required=False,
                    help='Number of items per batch. Default is 8.')
parser.add_argument('-w', '--num_workers', default=20, type=int, required=False,
                    help='Number of workers. Default is 20.')
args = parser.parse_args()

conjuntos = obtener_conjuntos(data_root='datos_preprocesados_EPISURG', n_conjuntos=args.num_folds, semilla=args.seed)
for i in range(args.num_folds):
    conjuntos[i] = mezclador_con_semilla(conjuntos[i], args.seed)

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

    if args.model_type == 'Classic':
        model = NNUNet3D_Classic(input_channels=1, num_classes=1)
        pl_model = SegmentationModule_Classic(model, learning_rate=1e-3, max_epochs=args.max_epochs)
        plot_callback = PlotCallback_Classic()
    if args.model_type == 'Hybrid':
        model = NNUNet3D_Hybrid(input_channels=1, num_classes=1, number_quantum_layers=5)
        pl_model = SegmentationModule_Hybrid(model, learning_rate=1e-3, max_epochs=args.max_epochs)
        plot_callback = PlotCallback_Hybrid()

    # Load the model and move it to GPU
    version = args.version
    n_checkpoint_files = 0
    for root, folders, files in os.walk(os.path.join('lightning_logs', f'version_{version}')):
        for file in files:
            if file.endswith('.ckpt'):
                checkpoint_path = os.path.join(root, file)
                n_checkpoint_files += 1
                assert n_checkpoint_files == 1, f'There is more than one weights file in folder {root}'
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

    filename_base = f"best_model-{model._get_name()}-n_epochs={args.max_epochs:02d}-"

    data_module = VideoDataModule(
        info_list=info_list,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_dice", # Metric to monitor
        mode="max",
        save_top_k=5, # Save the top 5 epochs
        filename=filename_base+"{epoch:02d}" # Using the full name causes issues when accessing checkpoints
    )


    # 3. Train
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        callbacks=[checkpoint_callback,
                early_stopping_callback,
                plot_callback,
                DeviceStatsMonitor()],
        logger=csv_logger,

        # Optimizations
        precision="16-mixed",       # Halves memory usage (approx.)
        accumulate_grad_batches=2,  # Simulates Batch Size: 2 * batch_size. Default 2 * 8 = 16
        
        # gradient clipping
        gradient_clip_val=1.0,          # Maximum allowed value
        gradient_clip_algorithm="norm", # "norm" clips based on the global norm of gradients; "value" clips by absolute value

        benchmark=True
    )
    trainer.fit(pl_model, data_module)


    # Run tests
    print("--- Training finished ---")
    print("--- Computing ensemble of the top 5 checkpoints... ---")

    # Read the actual files from disk
    ckpt_dir = checkpoint_callback.dirpath
    rutas_checkpoints = [
        os.path.join(ckpt_dir, f) 
        for f in os.listdir(ckpt_dir) 
        if f.endswith('.ckpt')
    ]

    # Initialize a new empty state_dict
    ensemble_state_dict = None
    num_checkpoints = len(rutas_checkpoints)

    for ruta in rutas_checkpoints:
        # Load the checkpoint
        ckpt = torch.load(ruta, map_location=torch.device('cpu'), weights_only=False)
        ckpt_state_dict = ckpt['state_dict']
        
        # If it is the first one, clone the structure dividing by the total number
        if ensemble_state_dict is None:
            ensemble_state_dict = {
                k: v.clone().float() / num_checkpoints 
                for k, v in ckpt_state_dict.items()
            }
        else:
            # Add the corresponding fraction of weights from subsequent checkpoints
            for k, v in ckpt_state_dict.items():
                ensemble_state_dict[k] += v.float() / num_checkpoints

    # Load the averaged weights into the Lightning model
    pl_model.load_state_dict(ensemble_state_dict)

    print("--- Averaged weights loaded. Running tests... ---")
    # Run the test using the model with the ensemble weights
    # Omit ckpt_path="best" to force using the weights just loaded into memory
    results = trainer.test(model=pl_model, datamodule=data_module)

    print("Test results (Weight Averaging):")
    print(results)

    # Save the results
    metrics = pd.read_csv(os.path.join(trainer.logger.log_dir, 'metrics.csv'))

    test_dice_avg  = [test_dice_avg for test_dice_avg in metrics['test_dice_avg'] if not np.isnan(test_dice_avg)][-1]
    test_dice_std  = [test_dice_std for test_dice_std in metrics['test_dice_std'] if not np.isnan(test_dice_std)][-1]
    test_dice_med  = [test_dice_med for test_dice_med in metrics['test_dice_med'] if not np.isnan(test_dice_med)][-1]
    test_dice_iqr  = [test_dice_iqr for test_dice_iqr in metrics['test_dice_iqr'] if not np.isnan(test_dice_iqr)][-1]
    test_loss  = [test_loss for test_loss in metrics['test_loss'] if not np.isnan(test_loss)][-1]

    version = trainer.logger.version

    # Save fine-tuning data
    os.makedirs(os.path.join('resultados', f'version_{version}'), exist_ok=True)
    ruta_resultados = os.path.join('resultados', f'version_{version}', 'resultados.csv')
    os.makedirs(os.path.dirname(ruta_resultados), exist_ok=True)

    # Create the CSV
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

# Save the boxplot of all data
os.makedirs(os.path.join('graficas', f'version_{versiones}'), exist_ok=True)
ruta_boxplot = os.path.join('graficas', f'version_{versiones}', 'test_Dice_boxplot.png')
os.makedirs(os.path.dirname(ruta_boxplot), exist_ok=True)

plt.figure()
plt.boxplot(total_test_dice, positions=[1])
plt.title('All test Dice coefficients')
plt.xticks([1], ['']) # Hide X-axis labels
plt.grid(True)
plt.savefig(ruta_boxplot)
plt.close()

# Save global fine-tuning data
avg_total_dice = np.mean(total_test_dice)
std_total_dice = np.std(total_test_dice)
med_total_dice = np.median(total_test_dice)
iqr_total_dice = np.quantile(total_test_dice, 0.75) - np.quantile(total_test_dice, 0.25)

os.makedirs(os.path.join('resultados', f'version_{versiones}'), exist_ok=True)
ruta_resultados = os.path.join('resultados', f'version_{versiones}', 'resultados_globales.csv')
os.makedirs(os.path.dirname(ruta_resultados), exist_ok=True)

# Create the CSV
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

# Save individual Dice coefficients
os.makedirs(os.path.join('resultados_enteros', f'version_{versiones}'), exist_ok=True)
ruta_resultados_enteros = os.path.join('resultados_enteros', f'version_{versiones}', f'fine-tuning_{args.model_type}_{versiones}.csv')
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

# Create the CSV
datos_dict = {
    'Archivo': rutas_test,
    'Coeficiente_de_Dice': total_test_dice,
    'Iteracion': iteraciones_pertenece
}
df = pd.DataFrame(datos_dict)
df.to_csv(ruta_resultados_enteros, index=False, encoding='utf-8')
