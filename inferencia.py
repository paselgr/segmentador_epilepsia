import torch
import os
from monai.inferers import SlidingWindowInferer
from modelos import NNUNet3D_Clasico, SegmentationModule_Clasico, NNUNet3D_Cuantica, SegmentationModule_Cuantica
import ants
import numpy as np
from preprocesamiento import extraer_cerebro_ants
import tempfile
import argparse


def procesar_escaner(escaner):
    img_npy = escaner.numpy()
    mask_npy = (img_npy != 0).astype(np.float32)
    # Generamos una máscara del cerebro completo
    mask_ants = escaner.new_image_like(mask_npy)
    
    cropped_escaner = ants.crop_image(escaner, mask_ants)
    
    mask_ants = ants.crop_image(mask_ants, mask_ants)
    
    # Corrección del bias field (N4)
    corregido_bias_field_escaner = ants.n4_bias_field_correction(cropped_escaner, mask=mask_ants)
    
    # Resampling a 1.0 x 1.0 x 1.0 mm del escáner y la máscara
    spacing = (1.0, 1.0, 1.0)
    resampled_escaner = ants.resample_image(
        corregido_bias_field_escaner,
        spacing,
        use_voxels=False,
        interp_type=4
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

    return escaner_normalizado


def revertir_prediccion_al_original(prediccion, escaner_original):

    # Creamos una carpeta temporal para guardar y acceder al archivo.
    # Si no se realiza este paso el proceso falla.
    with tempfile.TemporaryDirectory() as tmp_dir:
        ruta_mascara_de_paso = os.path.join(tmp_dir, 'de_paso.nii.gz')
        ants.image_write(prediccion, ruta_mascara_de_paso)
        mascara = ants.image_read(ruta_mascara_de_paso)

    # Alinea la máscara a la geometría del escáner usando el vecino más cercano
    mascara_acoplada = ants.resample_image_to_target(
        image=mascara,
        target=escaner_original,
        interp_type="nearestNeighbor",
    )

    return mascara_acoplada


parser = argparse.ArgumentParser(description='Inferencia del modelo.')
parser.add_argument('--model_type', type=str, choices=['Clasico', 'Cuantica'], required=True,
                    help='Especifica qué tipo de modelo se realiza la inferencia (Clasico o Cuantica).')
parser.add_argument('-v', '--version', type=int, required=True,
                    help='Versión del modelo sobre el cual se realizará la inferencia.')
parser.add_argument('-s', '--brain_strip', default=True, type=bool, required=False,
                    help='Si se realiza una extracción cerebral del escáner (True or False). Por defecto True.')
args = parser.parse_args()

if args.model_type == 'Clasico':
    model = NNUNet3D_Clasico(input_channels=1, num_classes=1)
    pl_model = SegmentationModule_Clasico(model, learning_rate=1e-3, max_epochs=1000)
if args.model_type == 'Cuantica':
    model = NNUNet3D_Cuantica(input_channels=1, num_classes=1, number_quantum_layers=5)
    pl_model = SegmentationModule_Cuantica(model, learning_rate=1e-3, max_epochs=1000)

version = args.version
brain_strip = args.brain_strip

carpeta_checkpoints = os.path.join('lightning_logs', f'version_{version}', 'checkpoints')
rutas_checkpoints = os.listdir(carpeta_checkpoints)
ensemble_state_dict = None
num_checkpoints = len(rutas_checkpoints)

if num_checkpoints == 1:
    print('--- Cargando los pesos del modelo... ---')
else:
    print(f"--- Calculando ensemble de los {num_checkpoints} checkpoints... ---")

for checkpoint in rutas_checkpoints:
    ruta = os.path.join(carpeta_checkpoints, checkpoint)
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

new_state_dict = {}
for key, value in ensemble_state_dict.items():
    new_key = key.replace('model.', '', 1) if key.startswith('model.') else key
    new_state_dict[new_key] = value

# Cargar los pesos promediados en el modelo Lightning
model.load_state_dict(new_state_dict)

print('--- Pesos del modelo cargados ---')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
pl_model.eval()


# Parámetros de la inferencia
patch_size = (112, 131, 98)
sw_batch_size = 4
overlap = 0.9
sliding_window = SlidingWindowInferer(
    roi_size=patch_size,
    sw_batch_size=sw_batch_size,
    overlap=overlap,
    mode="gaussian"
)

carpeta_entrada = 'datos_entrada_inferencia'
carpeta_salida  = 'datos_salida_inferencia'
for escaner in os.listdir(carpeta_entrada):
    escaner_ruta = os.path.join(carpeta_entrada, escaner)
    escaner_ants = extraer_cerebro_ants(escaner_ruta) if brain_strip else ants.image_read(escaner_ruta)
    escaner_ants_preprocesado = procesar_escaner(escaner_ants)

    escaner_numpy = escaner_ants_preprocesado.numpy()
    escaner_torch = torch.from_numpy(escaner_numpy)
    escaner_torch = escaner_torch.unsqueeze(0).unsqueeze(0)
    escaner_torch = escaner_torch.float().to(device)

    with torch.no_grad(): # Desactivar gradientes para ahorrar memoria
        logits_salida = sliding_window(escaner_torch, pl_model)

        probabilidades = torch.sigmoid(logits_salida)
        mascara_predicha = (probabilidades > 0.5).to(torch.uint8)
    mascara_predicha = mascara_predicha.squeeze().squeeze().cpu().numpy()


    # Convertir la máscara predicha en un archivo de ANTs con la información del escáner original
    mascara_predicha_ants = escaner_ants_preprocesado.new_image_like(mascara_predicha)
    ants.copy_image_info(escaner_ants_preprocesado, mascara_predicha_ants)
    mascara_acoplada = revertir_prediccion_al_original(mascara_predicha_ants, escaner_ants)

    # Guardamos la máscara en la carpeta de salida
    ants.image_write(mascara_acoplada, os.path.join(carpeta_salida, 'predicted_mask_' + escaner))

    print(f'--- Predicción del escaner {escaner} finalizada ---')
print('--- Predicciones finalizadas ---')
