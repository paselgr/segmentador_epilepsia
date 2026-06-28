# U-Net segmentadora de máscaras de resección

Segmentador de máscaras de resección de pacientes con epilepsia. El objetivo de este proyecto es mejorar y desarrollar los segmentadores de máscaras de resección, así como ver los efectos de los algoritmos cuánticos en estos modelos. El modelo está basado en la [nnU-Net](https://github.com/mic-dkfz/nnunet) [1] y está desarrollado exclusivamente con Python.

## Requisitos Previos

El código se ejecuta con [Python], la versión 3.10.19 no causa conflictos en las librerías.

Se necesitará descargar los pesos del modelo de [SynthStrip](https://zenodo.org/records/16535634) [2].

Una vez descargado deberá colocarse en la carpeta principal del proyecto.

También se necesitará descargar los conjuntos de datos de [EPISURG](https://rdr.ucl.ac.uk/articles/dataset/EPISURG_a_dataset_of_postoperative_magnetic_resonance_images_MRI_for_quantitative_analysis_of_resection_neurosurgery_for_refractory_epilepsy/9996158?file=26153588) [3] y [BraTS-GLI](https://www.synapse.org/Synapse:syn53708249/wiki/627759) [4] siguiendo las instrucciones de las páginas web.
Una vez descargados deberán colocarse en la carpeta principal del proyecto.

Las carpetas descargadas deberían tener la siguiente organización interna:

```text
📁 segmentador_epilepsia/
├── 📁 BraTS-GLI/                                       # Carpeta de BraTS
│   ├── 📄 ...
│   ├── 📁 BraTS2024-BraTS-GLI-AdditionalTrainingData/  # Carpeta con datos para el entrenamiento
│   ├── 📁 BraTS2024-BraTS-GLI-TrainingData/            # Carpeta con datos para el entrenamiento
│   └── 📁 ...
├── 📁 EPISURG/
│   ├── 📁 subjects/                # Carpeta con datos para el fine-tuning
│   ├── 📄 README.md
│   ├── 📄 README.txt
│   └── 📄 subjects.csv             # Información sobre los datos de EPISURG
├── 📄 synthstrip.1.pt              # Pesos de SynthStrip [Descargados]
├── 📄 preprocesamiento.py          # Script de preparación de datos
├── 📄 entrenamiento.py             # Script de entrenamiento principal
├── 📄 fine-tuning.py               # Script para fine-tuning
├── 📄 requirements.txt             # Dependencias del proyecto
└── 📄 README.md                    # Documentación del proyecto


## Instalación

Sigue estos pasos paso a paso para configurar el entorno de desarrollo en tu máquina:

1. **Clona el repositorio** en tu máquina local:
   ```bash
   git clone [https://github.com/paselgr/segmentador_epilepsia.git](https://github.com/paselgr/segmentador_epilepsia.git)
   ```

2. **Navega** al directorio del proyecto:
   ```bash
   cd nombre-del-repo
   ```

3. **Instala las dependencias** y paquetes necesarios:

Deberás instalar PyTorch. Si tienes una GPU de NVIDIA compatible, instala la versión acelerada por hardware (CUDA 12.1):

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Si no dispones de una tarjeta gráfica NVIDIA, instala la versión básica. Es más ligera pero los entrenamientos serán notablemente más lentos:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

Una vez instalado PyTorch, instala el resto de librarías mediante

```bash
pip install -r requirements.txt
```

## Uso

Una vez que todas las librerías estén instaladas, de deberán preprocesar los conjuntos de datos mediante

```bash
python preprocesamiento.py --dataset nombre_del_conjunto_de_datos
```

Cuando los datos estén preprocesados puedes entrenar el modelo mediante

```bash
python entrenamiento.py --model_type tipo_de_modelo
```

y fine-tuning mediante

```bash
python fine-tuning.py --model_type tipo_de_modelo --version numero_version
```

## Referencias

[1] nnU-Net:
Si utilizas o te basas en la arquitectura de este proyecto, por favor considera citar el trabajo original de nnU-Net:

Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.

[2] SynthStrip:
fepegar, “B-spline deformation.ipynb.” GitHub Gist. (https://gist.github.com/
fepegar/b723d15de620cd2a3a4dbd71e491b59d)

[3] EPISURG:
Pérez-García, F., Rodionov, R., Alim-Marvasti, A., Sparks, R., Duncan, J., & Ourselin, S. (2020). EPISURG: a dataset of postoperative magnetic resonance images (MRI) for quantitative analysis of resection neurosurgery for refractory epilepsy. University College London. DOI, 1(0.5522), 04.

Pérez-García, F., Dorent, R., Rizzi, M., Cardinale, F., Frazzini, V., Navarro, V., ... & Ourselin, S. (2021). A self-supervised learning strategy for postoperative brain cavity segmentation simulating resections. International Journal of Computer Assisted Radiology and Surgery, 16(10), 1653-1661.

[4] BraTS-GLI:
de Verdier, M. C., Saluja, R., Gagnon, L., LaBella, D., Baid, U., Tahon, N. H., ... & Rudie, J. D. (2024). The 2024 brain tumor segmentation (brats) challenge: Glioma segmentation on post-treatment mri. arXiv preprint arXiv:2405.18368.
