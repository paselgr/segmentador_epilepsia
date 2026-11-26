import argparse
import os
import tempfile

import ants
from monai.inferers import SlidingWindowInferer
import numpy as np
import torch

from models import (
    NNUNet3D_Classic,
    NNUNet3D_Hybrid,
    SegmentationModule_Classic,
    SegmentationModule_Hybrid,
)
from preprocessing import extraer_cerebro_ants


def procesar_escaner(escaner):
    img_npy = escaner.numpy()
    mask_npy = (img_npy != 0).astype(np.float32)
    # Generate a whole-brain mask
    mask_ants = escaner.new_image_like(mask_npy)
    
    cropped_escaner = ants.crop_image(escaner, mask_ants)
    
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
    mask_ants = ants.resample_image(
        mask_ants,
        spacing,
        use_voxels=False,
        interp_type=1
    )

    # Normalize non-background data using Z-score
    voxeles_validos = resampled_escaner[mask_ants == 1]
    # Compute mean and standard deviation and normalize
    mean = voxeles_validos.mean()
    std  = voxeles_validos.std()
    escaner_normalizado = (resampled_escaner - mean) / std
    escaner_normalizado = escaner_normalizado * mask_ants

    return escaner_normalizado


def revertir_prediccion_al_original(prediccion, escaner_original):

    # Create a temporary folder to save and access the file.
    # Without this step, the process fails.
    with tempfile.TemporaryDirectory() as tmp_dir:
        ruta_mascara_de_paso = os.path.join(tmp_dir, 'de_paso.nii.gz')
        ants.image_write(prediccion, ruta_mascara_de_paso)
        mascara = ants.image_read(ruta_mascara_de_paso)

    # Align the mask to the scan's geometry using nearest-neighbor interpolation
    mascara_acoplada = ants.resample_image_to_target(
        image=mascara,
        target=escaner_original,
        interp_type="nearestNeighbor",
    )

    return mascara_acoplada


parser = argparse.ArgumentParser(description='Model inference.')
parser.add_argument('--model_type', type=str, choices=['Classic', 'Hybrid'], required=True,
                    help='Specifies which model type to run inference (Classic or Hybrid).')
parser.add_argument('-v', '--version', type=int, required=True,
                    help='Version of the model to run inference on.')
parser.add_argument('-s', '--brain_strip', default=True, type=bool, required=False,
                    help='Whether to perform brain extraction on the scan (True or False). Default is True.')
args = parser.parse_args()

if args.model_type == 'Classic':
    model = NNUNet3D_Classic(input_channels=1, num_classes=1)
    pl_model = SegmentationModule_Classic(model, learning_rate=1e-3, max_epochs=1000)
if args.model_type == 'Hybrid':
    model = NNUNet3D_Hybrid(input_channels=1, num_classes=1, number_quantum_layers=5)
    pl_model = SegmentationModule_Hybrid(model, learning_rate=1e-3, max_epochs=1000)

version = args.version
brain_strip = args.brain_strip

carpeta_checkpoints = os.path.join('lightning_logs', f'version_{version}', 'checkpoints')
rutas_checkpoints = os.listdir(carpeta_checkpoints)
ensemble_state_dict = None
num_checkpoints = len(rutas_checkpoints)

if num_checkpoints == 1:
    print('--- Loading model weights... ---')
else:
    print(f"--- Computing ensemble of {num_checkpoints} checkpoints... ---")

for checkpoint in rutas_checkpoints:
    ruta = os.path.join(carpeta_checkpoints, checkpoint)
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

new_state_dict = {}
for key, value in ensemble_state_dict.items():
    new_key = key.replace('model.', '', 1) if key.startswith('model.') else key
    new_state_dict[new_key] = value

# Load the averaged weights into the Lightning model
model.load_state_dict(new_state_dict)

print('--- Model weights loaded ---')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
pl_model.eval()


# Inference parameters
patch_size = (112, 131, 98)
sw_batch_size = 4
overlap = 0.9
sliding_window = SlidingWindowInferer(
    roi_size=patch_size,
    sw_batch_size=sw_batch_size,
    overlap=overlap,
    mode="gaussian"
)

carpeta_entrada = 'input_data_inference'
carpeta_salida  = 'output_data_inference'
for escaner in os.listdir(carpeta_entrada):
    escaner_ruta = os.path.join(carpeta_entrada, escaner)
    escaner_ants = extraer_cerebro_ants(escaner_ruta) if brain_strip else ants.image_read(escaner_ruta)
    escaner_ants_preprocesado = procesar_escaner(escaner_ants)

    escaner_numpy = escaner_ants_preprocesado.numpy()
    escaner_torch = torch.from_numpy(escaner_numpy)
    escaner_torch = escaner_torch.unsqueeze(0).unsqueeze(0)
    escaner_torch = escaner_torch.float().to(device)

    with torch.no_grad(): # Disable gradients to save memory
        logits_salida = sliding_window(escaner_torch, pl_model)

        probabilidades = torch.sigmoid(logits_salida)
        mascara_predicha = (probabilidades > 0.5).to(torch.uint8)
    mascara_predicha = mascara_predicha.squeeze().squeeze().cpu().numpy()


    # Convert the predicted mask into an ANTs image with the original scan's metadata
    mascara_predicha_ants = escaner_ants_preprocesado.new_image_like(mascara_predicha)
    ants.copy_image_info(escaner_ants_preprocesado, mascara_predicha_ants)
    mascara_acoplada = revertir_prediccion_al_original(mascara_predicha_ants, escaner_ants)

    # Save the mask to the output folder
    ants.image_write(mascara_acoplada, os.path.join(carpeta_salida, 'predicted_mask_' + escaner))

    print(f'--- Scan {escaner} prediction finished ---')
print('--- Predictions finished ---')
