import argparse
import os
import subprocess
import sys
import tempfile

import ants
import numpy as np


def filtrar_archivos_mod_mascara_BraTS(main_root, folder, list_of_files):
    for file in sorted(list_of_files, key=lambda x: '-seg' in x, reverse=True): # Place the file containing '-seg' first in the list
        file_source_path = os.path.join(main_root, folder, file)
        if file.endswith('-seg.nii.gz'):
            ruta_mascara = file
            ants_mask = ants.image_read(file_source_path)
            array_mask = ants_mask.numpy()
            if not np.any(np.isin(array_mask, [1, 3, 4])): # If neither 1, nor 3, nor 4 are in the array, skip to the next folder
                return None
            # Modify the original mask
            array_binario = np.isin(array_mask, [1, 3, 4]).astype(np.float32)
            mascara = ants_mask.new_image_like(array_binario)
        if file.endswith('-t1n.nii.gz'):
            ruta_escaner = file
            escaner = ants.image_read(file_source_path)
    return escaner, mascara, ruta_escaner, ruta_mascara

def procesar(escaner, mascara):
    img_npy = escaner.numpy()
    mask_npy = (img_npy != 0).astype(np.float32)
    # Generate a whole-brain mask
    mask_ants = escaner.new_image_like(mask_npy)
    
    cropped_escaner = ants.crop_image(escaner, mask_ants)
    cropped_mascara = ants.crop_image(mascara, mask_ants)
    
    mask_ants = ants.crop_image(mask_ants, mask_ants)
    
    # Bias field correction (N4)
    corregido_bias_field_escaner = ants.n4_bias_field_correction(cropped_escaner, mask=mask_ants)
    
    # Resample scan and mask to 1.0 x 1.0 x 1.0 mm
    spacing = (1.0, 1.0, 1.0)
    resampled_escaner = ants.resample_image(
        corregido_bias_field_escaner,
        spacing,
        use_voxels=False,
        interp_type=4
    )
    resampled_mascara = ants.resample_image(
        cropped_mascara,
        spacing,
        use_voxels=False,
        interp_type=1
    )
    mask_ants = ants.resample_image(
        mask_ants,
        spacing,
        use_voxels=False,
        interp_type=1
    )

    # Z-score normalize non-background data
    voxeles_validos = resampled_escaner[mask_ants == 1]
    # Compute mean and standard deviation and normalize
    mean = voxeles_validos.mean()
    std  = voxeles_validos.std()
    escaner_normalizado = (resampled_escaner - mean) / std
    escaner_normalizado = escaner_normalizado * mask_ants

    return escaner_normalizado, resampled_mascara

def extraer_cerebro_ants(ruta_nifti_entrada):
    """
    Takes a NIfTI file path, runs nipreps-synthstrip, and returns
    the resulting image as an ANTs object in memory.
    """

    NOMBRE_EJECUTABLE = "nipreps-synthstrip" 
    python_executable_path = sys.executable
    env_bin_path = os.path.dirname(python_executable_path)
    executable_path = os.path.join(env_bin_path, NOMBRE_EJECUTABLE)
    
    # Get the path to the synthstrip.1.pt file
    directorio_script = os.path.dirname(os.path.abspath(__file__))
    modelo_path = os.path.join(directorio_script, 'synthstrip.1.pt')

    # Verify that the input exists
    if not os.path.exists(ruta_nifti_entrada):
        raise FileNotFoundError(f"Could not find the NIfTI file with path: {ruta_nifti_entrada}")

    # Create a temporary directory so the output is a NIfTI file read with ANTs
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Define the temporary output path
        ruta_salida_temporal = os.path.join(temp_dir, 'brain_temp.nii.gz')

        comando = [
            executable_path,
            '-i', ruta_nifti_entrada,
            '-o', ruta_salida_temporal,
            '--model', modelo_path 
        ]

        try:
            subprocess.run(comando, check=True, capture_output=True, text=True)
            
            # Read the temporary file before deleting it
            imagen_ants = ants.image_read(ruta_salida_temporal)
            
            return imagen_ants

        except subprocess.CalledProcessError as e:
            print(f"ERROR: Processing failed (exit code {e.returncode}).")
            print("\n--- STDERR ERROR ---")
            print(e.stderr)
            return None
        except FileNotFoundError:
            print(f"ERROR: Could not find executable at: {executable_path}")
            return None

def main():
    parser = argparse.ArgumentParser(description='Processing NIfTI scans with SynthStrip and ANTs.')
    parser.add_argument('--dataset', type=str, required=True, choices=['BraTS', 'EPISURG'],
                        help='Specify which dataset to process (BraTS or EPISURG)')
    args = parser.parse_args()

    dataset = args.dataset
    if dataset == 'BraTS':
        main_exit_path = 'datos_preprocesados_BraTS'
        try:
            os.mkdir(main_exit_path)
        except FileExistsError:
            sys.exit(f'La carpeta {main_exit_path} ya existe. Elimínela para ejecutar el script.')
        main_root_train     = os.path.join('BraTS-GLI', 'BraTS2024-BraTS-GLI-TrainingData', 'training_data1_v2')
        main_root_add_train = os.path.join('BraTS-GLI', 'BraTS2024-BraTS-GLI-AdditionalTrainingData', 'training_data_additional')
        main_roots = [main_root_train, main_root_add_train]
        for main_root in main_roots:
            for folder in os.listdir(main_root):
                folder_path = os.path.join(main_root, folder)
                list_of_files = os.listdir(folder_path)
                if not any(['-seg' in x for x in list_of_files]):
                    # Every folder should contain a mask file (containing -seg)
                    print(f'Folder {folder} does not have any file with \'-seg\'')
                    continue
                salida_filtrado = filtrar_archivos_mod_mascara_BraTS(main_root, folder, list_of_files)
                if salida_filtrado is None:
                    continue
                escaner, mascara, ruta_escaner, ruta_mascara = salida_filtrado
                escaner_procesado, mascara_procesada = procesar(escaner, mascara)

                os.mkdir(os.path.join(main_exit_path, folder))
                ruta_escaner_procesado = os.path.join(main_exit_path, folder, ruta_escaner)
                ruta_mascara_procesada = os.path.join(main_exit_path, folder, ruta_mascara)
                ants.image_write(escaner_procesado, ruta_escaner_procesado)
                ants.image_write(mascara_procesada, ruta_mascara_procesada)
    elif dataset == 'EPISURG':
        main_exit_path = 'datos_preprocesados_EPISURG'
        try:
            os.mkdir(main_exit_path)
        except FileExistsError:
            sys.exit(f'La carpeta {main_exit_path} ya existe. Elimínela para ejecutar el script.')
        main_root = os.path.join('EPISURG', 'subjects')
        for folder in os.listdir(main_root):
            postop_folder = os.path.join(main_root, folder, 'postop')
            if any([file.endswith('-seg-1.nii.gz') for file in os.listdir(postop_folder)]):
                for file in os.listdir(postop_folder):
                    if file.endswith('-t1mri-1.nii.gz'):
                        ruta_escaner = os.path.join(postop_folder, file)
                    if file.endswith('-seg-1.nii.gz'):
                        ruta_mascara = os.path.join(postop_folder, file)
                        mascara = ants.image_read(ruta_mascara)
                escaner_synthstrip = extraer_cerebro_ants(ruta_escaner)
                # If conditions are not met, extraer_cerebro_ants returns None and we move to the next file
                if escaner_synthstrip is None:
                    continue
                escaner_procesado, mascara_procesada = procesar(escaner_synthstrip, mascara)
                os.mkdir(os.path.join(main_exit_path, folder))
                ruta_escaner_procesado = os.path.join(main_exit_path, folder, os.path.basename(ruta_escaner))
                ruta_mascara_procesada = os.path.join(main_exit_path, folder, os.path.basename(ruta_mascara))
                ants.image_write(escaner_procesado, ruta_escaner_procesado)
                ants.image_write(mascara_procesada, ruta_mascara_procesada)
    else:
        raise ValueError('Invalid option. Valid options are \'BraTS\' and \'EPISURG\'.')
    print('--- Preprocessing finished ---')

if __name__ == '__main__':
    main()
