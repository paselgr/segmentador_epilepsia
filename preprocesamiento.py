import argparse
import os
import subprocess
import sys
import tempfile

import ants
import numpy as np


def filtrar_archivos_mod_mascara_BraTS(main_root, folder, list_of_files):
    for file in sorted(list_of_files, key=lambda x: '-seg' in x, reverse=True): # Coloco el archivo que contiene '-seg' el primero en la lista
        file_source_path = os.path.join(main_root, folder, file)
        if file.endswith('-seg.nii.gz'):
            ruta_mascara = file
            ants_mask = ants.image_read(file_source_path)
            array_mask = ants_mask.numpy()
            if not np.any(np.isin(array_mask, [1, 3, 4])): # Si ni 1, ni 3, ni 4 están en el array, salta a la siguiente carpeta
                return None
            # Modificamos la máscara original
            array_binario = np.isin(array_mask, [1, 3, 4]).astype(np.float32)
            mascara = ants_mask.new_image_like(array_binario)
        if file.endswith('-t1n.nii.gz'):
            ruta_escaner = file
            escaner = ants.image_read(file_source_path)
    return escaner, mascara, ruta_escaner, ruta_mascara

def procesar(escaner, mascara):
    img_npy = escaner.numpy()
    mask_npy = (img_npy != 0).astype(np.float32)
    # Generamos una máscara del cerebro completo
    mask_ants = escaner.new_image_like(mask_npy)
    
    cropped_escaner = ants.crop_image(escaner, mask_ants)
    cropped_mascara = ants.crop_image(mascara, mask_ants)
    
    mask_ants = ants.crop_image(mask_ants, mask_ants)
    
    # Corrección del bias field (N4)
    # mask_para_n4 = cropped_escaner > 0
    corregido_bias_field_escaner = ants.n4_bias_field_correction(cropped_escaner, mask=mask_ants)
    
    # Resampling a 1.0 x 1.0 x 1.0 mm del escáner y la máscara
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

    # Normalizamos con Z_score los datos que no sean fondo
    voxeles_validos = resampled_escaner[mask_ants == 1]
    # Calculamos la media y la desviación estándar y normalizamos
    mean = voxeles_validos.mean()
    std  = voxeles_validos.std()
    escaner_normalizado = (resampled_escaner - mean) / std
    escaner_normalizado = escaner_normalizado * mask_ants

    return escaner_normalizado, resampled_mascara

def extraer_cerebro_ants(ruta_nifti_entrada):
    """
    Toma la ruta de un NIfTI, ejecuta nipreps-synthstrip y devuelve
    la imagen resultante como un objeto ANTs en memoria.
    """

    NOMBRE_EJECUTABLE = "nipreps-synthstrip" 
    python_executable_path = sys.executable
    env_bin_path = os.path.dirname(python_executable_path)
    executable_path = os.path.join(env_bin_path, NOMBRE_EJECUTABLE)
    
    # Obtenemos la ruta del archivo synthstrip.1.pt
    directorio_script = os.path.dirname(os.path.abspath(__file__))
    modelo_path = os.path.join(directorio_script, 'synthstrip.1.pt')

    # Verificamos que la entrada existe
    if not os.path.exists(ruta_nifti_entrada):
        raise FileNotFoundError(f"No se encontró el archivo NIfTI de entrada en: {ruta_nifti_entrada}")

    # Creamos un directorio temporal para leerlo y que la salida sea un archivo NIfTI leído con ANTs
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Definimos la ruta temporal
        ruta_salida_temporal = os.path.join(temp_dir, 'brain_temp.nii.gz')

        comando = [
            executable_path,
            '-i', ruta_nifti_entrada,
            '-o', ruta_salida_temporal,
            '--model', modelo_path 
        ]

        try:
            subprocess.run(comando, check=True, capture_output=True, text=True)
            
            # Leemos el archivo temporal antes de eliminarlo
            imagen_ants = ants.image_read(ruta_salida_temporal)
            
            return imagen_ants

        except subprocess.CalledProcessError as e:
            print(f"ERROR: El procesamiento falló (código de salida {e.returncode}).")
            print("\n--- ERROR DE STDERR ---")
            print(e.stderr)
            return None
        except FileNotFoundError:
            print(f"ERROR: No se pudo encontrar el ejecutable en: {executable_path}")
            print("Esto es extraño. Vuelve a ejecutar 'pip install .[pydra]' en la carpeta synthstrip-main.")
            return None

def main():
    parser = argparse.ArgumentParser(description='Procesamiento de escáneres NIfTI con SynthStrip y ANTs.')
    parser.add_argument('--dataset', type=str, required=True, choices=['BraTS', 'EPISURG'],
                        help='Especifica qué dataset quieres procesar (BraTS o EPISURG)')
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
                    # Todos las carpetas deberían contener un archivo con la máscara (que contenga -seg)
                    print(f'La carpeta {folder} no tiene ningún archivo con \'-seg\'')
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
    elif dataset == 'EPISURG': ### argparse
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
                # Si no cumple las extraer_cerebro_ants devuelve None y pasamos al siguiente archivo
                if escaner_synthstrip is None:
                    continue
                escaner_procesado, mascara_procesada = procesar(escaner_synthstrip, mascara)
                os.mkdir(os.path.join(main_exit_path, folder))
                ruta_escaner_procesado = os.path.join(main_exit_path, folder, os.path.basename(ruta_escaner))
                ruta_mascara_procesada = os.path.join(main_exit_path, folder, os.path.basename(ruta_mascara))
                ants.image_write(escaner_procesado, ruta_escaner_procesado)
                ants.image_write(mascara_procesada, ruta_mascara_procesada)
    else:
        raise ValueError('Opción no válida. Las opciones válidas son \'BraTS\' y \'EPISURG\'.')
    print('--- Preprocesamiento finalizado ---')

if __name__ == '__main__':
    main()
