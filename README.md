# Resection Mask Segmentation U-Net

Resection mask segmenter for patients with epilepsy. The aim of this project is to improve and develop resection mask segmenters, as well as to observe the effects of quantum algorithms on these models. The model is based on [nnU-Net](https://github.com/mic-dkfz/nnunet) [1] and is developed exclusively in Python.

## Prerequisites

The code runs with [Python]. Version 3.10.19 does not cause library conflicts.

You will need to download the model weights from [SynthStrip](https://zenodo.org/records/16535634) [2].

Once downloaded, the file must be placed in the root folder of the project.

You will also need to download the [EPISURG](https://rdr.ucl.ac.uk/articles/dataset/EPISURG_a_dataset_of_postoperative_magnetic_resonance_images_MRI_for_quantitative_analysis_of_resection_neurosurgery_for_refractory_epilepsy/9996158?file=26153588) [3] and [BraTS-GLI](https://www.synapse.org/Synapse:syn53708249/wiki/627759) [4] datasets following the instructions on their websites.
Once downloaded, they must be placed in the root folder of the project.

The downloaded folders should have the following internal organization:

```text
📁 segmentador_epilepsia/
├── 📁 BraTS-GLI/                                        # BraTS folder
│   ├── 📄 ...
│   ├── 📁 BraTS2024-BraTS-GLI-AdditionalTrainingData/   # Folder with the training data
│   ├── 📁 BraTS2024-BraTS-GLI-TrainingData/             # Folder with the training data
│   └── 📁 ...
├── 📁 EPISURG/                                          # EPISURG folder
│   ├── 📁 subjects/                                     # Folder with the fine-tuning data
│   ├── 📄 README.md
│   ├── 📄 README.txt
│   └── 📄 subjects.csv         # Informarion about the EPISURG data
├── 📁 input_data_inference/    # Input folder for the inference 
├── 📁 output_data_inference/   # Output folder for the inference 
├── 📄 LICENSE                  # The repository license
├── 📄 README.md                # Project documentation
├── 📄 Results_fine-tuning_Classic.csv      # Classic model results
├── 📄 Results_fine-tuning_Hybrid.csv       # Hybrid model results
├── 📄 fine-tuning.py           # Fine-tuning script
├── 📄 inference.py             # Inference script
├── 📄 models.py                # Script with the models
├── 📄 preprocessing.py         # Preprocessing script
├── 📄 requirements.txt         # Project dependencies
└── 📄 training.py              # Training script
```

## Installation

Follow these step-by-step instructions to set up the development environment on your machine:

1. **Clone the repository** to your local machine:
   ```bash
   git clone [https://github.com/paselgr/segmentador_epilepsia.git](https://github.com/paselgr/segmentador_epilepsia.git)
   ```

2. **Navigate** to the project directory:
   ```bash
   cd segmentador_epilepsia
   ```

3. **Install the dependencies** and required packages:

You will need to install PyTorch. If you have a compatible NVIDIA GPU, install the hardware-accelerated version (CUDA 12.1):

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

If you do not have an NVIDIA graphics card, install the basic version. It is lighter, though training will be noticeably slower:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

Once PyTorch is installed, install the remaining libraries using

```bash
pip install -r requirements.txt
```

## Usage

Once all libraries are installed, both datasets must be preprocessed using

```bash
python preprocessing.py --dataset dataset_name
```

When the data is preprocessed, you can train the model using

```bash
python training.py --model_type model_type_name
```

and fine-tune it using

```bash
python fine-tuning.py --model_type model_type_name --version version_number
```

Inference is run on all files in the `input_data_inference` folder with

```bash
python inference.py --model_type model_type_name --version version_number --brain_strip bool_brain_strip
```

## Note

The files [`Results_fine-tuning_Classic.csv`](./Results_fine-tuning_Classic.csv) and [`Results_fine-tuning_Hybrid.csv`](./Results_fine-tuning_Hybrid.csv) are included in the folder with all test coefficient data from the model after training and fine-tuning with default parameters in the classical model and quantum version, respectively, in case the user is interested in computing other metrics not included here.

## References

[1] nnU-Net:
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.

[2] SynthStrip:
fepegar, “B-spline deformation.ipynb.” GitHub Gist. (https://gist.github.com/fepegar/b723d15de620cd2a3a4dbd71e491b59d)

[3] EPISURG:
Pérez-García, F., Rodionov, R., Alim-Marvasti, A., Sparks, R., Duncan, J., & Ourselin, S. (2020). EPISURG: a dataset of postoperative magnetic resonance images (MRI) for quantitative analysis of resection neurosurgery for refractory epilepsy. University College London. DOI, 1(0.5522), 04.

Pérez-García, F., Dorent, R., Rizzi, M., Cardinale, F., Frazzini, V., Navarro, V., ... & Ourselin, S. (2021). A self-supervised learning strategy for postoperative brain cavity segmentation simulating resections. International Journal of Computer Assisted Radiology and Surgery, 16(10), 1653-1661.

[4] BraTS-GLI:
de Verdier, M. C., Saluja, R., Gagnon, L., LaBella, D., Baid, U., Tahon, N. H., ... & Rudie, J. D. (2024). The 2024 brain tumor segmentation (brats) challenge: Glioma segmentation on post-treatment mri. arXiv preprint arXiv:2405.18368.
