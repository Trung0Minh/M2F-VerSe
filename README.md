# Mask2Former on VerSe: Vertebrae Segmentation

This repository contains a robust refactoring of the **Mask2Former** architecture specifically fine-tuned for the **VerSe (Large Scale Vertebrae Segmentation Challenge)** dataset. 

Our goal is to seamlessly bridge the gap between 3D medical imaging (CT scans) and state-of-the-art 2D panoptic/instance segmentation models by providing a complete pipeline: from 3D-to-2D preprocessing to training, evaluation, and visualization.

---

## 📌 Acknowledgements & Citations

This project is built upon the incredible work of two distinct research efforts. If you use this repository, please ensure you cite the original authors:

### 1. The VerSe Dataset
The 3D CT data and original preprocessing concepts are derived from the VerSe project.
* **Official Repo:** [https://github.com/anjany/verse](https://github.com/anjany/verse)
* **Citation:** Sekuboyina A et al., *VerSe: A Vertebrae Labelling and Segmentation Benchmark for Multi-detector CT Images*, 2021. ([Paper](https://doi.org/10.1016/j.media.2021.102166))

### 2. Mask2Former Architecture
The core deep learning architecture and training loop are derived from Mask2Former.
* **Official Repo:** [https://github.com/facebookresearch/Mask2Former](https://github.com/facebookresearch/Mask2Former)
* **Citation:** Cheng, B., et al. *Masked-attention Mask and Universal Image Segmentation.* CVPR 2022.

---

## 📂 Folder Structure

The repository is organized to separate data preprocessing from model training:

```text
M2F-VerSe/
├── data/                      # Place your downloaded 3D VerSe raw nifti files here
├── dataset_verse_2d/          # The output folder where 2D slices and JSON metadata are saved after processing
├── utils/                     # 3D-to-2D Preprocessing Pipeline
│   ├── process_dataset.ipynb  # Main notebook to convert 3D NIfTI -> 2D PNGs + JSON
│   ├── test_sample.ipynb      # Quick sandbox to test 2D conversion on a single subject
│   └── sample/                # A single 3D subject used by test_sample.ipynb
└── Mask2Former/               # The core Segmentation Framework
    ├── configs/verse/         # Universal YAML configs (Instance, Panoptic, Semantic)
    ├── mask2former/           # Model architecture and Universal Dataset Registration
    ├── weights/               # Downloaded pre-trained weights (e.g., from COCO/ADE20k)
    ├── results/               # Generated visualization outputs from scripts
    └── visualization/         # Custom scripts to visualize predictions vs. Ground Truth
```

---

## 🚀 Getting Started

### 1. Environment Setup
We recommend using Python 3.10+ and PyTorch with CUDA support.

```bash
# Install PyTorch (Modify to match your CUDA version)
# See: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Detectron2 (Requires PyTorch to be installed first)
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation

# Install PanopticAPI
python -m pip install 'git+https://github.com/cocodataset/panopticapi.git'

# Install project dependencies
pip install -r requirements.txt
```

#### 🛠️ Compiling CUDA Operators (Required)
Mask2Former uses custom CUDA operators for Multi-Scale Deformable Attention. **This step must succeed before training.**

To avoid "Version Mismatch" or "Undefined Symbol" errors, follow these steps:
1.  **Align CUDA Versions:** Your system's `nvcc` version **MUST** match your PyTorch CUDA version (e.g., if you installed PyTorch for CUDA 12.1, you need CUDA Toolkit 12.1).
    ```bash
    nvcc --version
    python -c "import torch; print(torch.version.cuda)"
    ```

**For Conda Users (Recommended if you lack system-wide CUDA/GCC):**
If you have version mismatches or lack a compatible compiler (CUDA 12.1 needs **GCC 12** or lower), you can install them directly into your environment:
```bash
# Install compatible CUDA 12.1 Toolkit and GCC 12
conda install -c nvidia cuda-toolkit=12.1 cuda-nvcc=12.1 cuda-cccl=12.1 \
    libcusparse-dev=12.1 libcublas-dev=12.1 libcufft-dev=11.0 \
    libcurand-dev=10.3 libcusolver-dev=11.4 -y
conda install -c conda-forge gcc_linux-64=12 gxx_linux-64=12 -y

# Set path and compile (Conda paths are now auto-detected by setup.py)
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export FORCE_CUDA=1
cd Mask2Former/mask2former/modeling/pixel_decoder/ops
rm -rf build *.egg-info
sh make.sh
```

**Standard Compilation:**
2.  **Set Environment Variables:**
    ```bash
    export CUDA_HOME=/usr/local/cuda  # Path to your matching CUDA toolkit
    export PATH=$CUDA_HOME/bin:$PATH
    export FORCE_CUDA=1
    ```
3.  **Compile:**
    ```bash
    cd Mask2Former/mask2former/modeling/pixel_decoder/ops
    rm -rf build *.egg-info  # Always start with a clean build
    sh make.sh
    ```

### 2. Data Preprocessing (3D to 2D)
Before training, you must convert the 3D NIfTI volumes into 2D slices.

1. **Download & Restructure the Dataset:**
   - Download the raw VerSe dataset from this [Google Drive link](https://drive.google.com/drive/folders/1HkjTonPx4Ei4YRDBKHYWG-SF9gevr6w_?usp=drive_link).
   - The original dataset does not follow the required structure, so you **must** manually restructure it into the `data/` folder as follows for the preprocessing scripts to work:
     ```text
     data/
     ├── train/
     │   ├── derivatives/
     │   └── rawdata/
     ├── val/                                                                                 
     │   ├── derivatives/                                                                     
     │   └── rawdata/                                                                         
     └── test/                                                                                
         ├── derivatives/                                                                     
         └── rawdata/                                                                         
     ```
     *Note: Each subject folder (e.g., `sub-verse500`) should be placed inside its respective `rawdata` (for CT scans) or `derivatives` (for segmentation masks) subfolder.*

2. **Run Pipeline:** Open a terminal and launch Jupyter:
   ```bash
   jupyter notebook
   ```
   Execute `utils/process_dataset.ipynb` to process the entire dataset. 
   - **Important:** By default, the notebook uses `EXPORT_STYLE = 'ade20k'`, which populates the `dataset_verse_2d/ade20k/` directory. If you change this style, ensure your YAML configs match the new path.
3. **Verify Conversion:** You can use `utils/test_sample.ipynb` to verify the bone windowing and slice extraction on a sample subject.

### 3. Model Configuration & Weights
The configuration files are located in `Mask2Former/configs/verse/`. They are set up to use a ResNet-50 backbone by default.
1. Download pre-trained weights from the [Mask2Former Model Zoo](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md).
2. Place them in `Mask2Former/weights/`. 
   - For `verse_panoptic_R50.yaml`, ensure the file is named `ade20k_panoptic_R50.pkl` (or update the `MODEL.WEIGHTS` path in the YAML).
3. **Custom Backbones (Optional):** If you use a different backbone (e.g., Swin Transformer):
   - **Weight Conversion:** Convert official `.pth` weights to `.pkl` using `python tools/convert-pretrained-swin-model-to-d2.py path/to/model.pth weights/model.pkl`.
   - **Architecture Matching:** Ensure your YAML config parameters (`EMBED_DIM`, `DEPTHS`, etc.) precisely match the architecture of the backbone you downloaded.
4. Ensure the `MODEL.WEIGHTS` path in your chosen config file (`verse_panoptic_R50.yaml`) points to your weight file.

---

## 🏃‍♂️ Training & Evaluation

### Training
Navigate to the `Mask2Former` directory and run the training script. Detectron2 handles the universal registration dynamically based on the `DATASETS.VERSE_ROOT` path in your YAML.

```bash
cd Mask2Former
python train.py --num-gpus 1 \
  --config-file configs/verse/verse_panoptic_R50.yaml
```

### Evaluation
To evaluate a trained model, append the `--eval-only` flag and point to your trained weights:

```bash
python train.py --num-gpus 1 \
  --config-file configs/verse/verse_panoptic_R50.yaml \
  --eval-only MODEL.WEIGHTS output/verse_panoptic_R50/model_final.pth
```

---

## 👁️ Visualization

We provide robust, command-line friendly visualization tools specifically tailored for the VerSe dataset. These scripts automatically compare your model's predictions side-by-side with the manual Ground Truth annotations.

Run these from the `Mask2Former` directory:

**Instance Segmentation:**
```bash
python visualization/visualize_instance.py --config configs/verse/verse_instance_R50.yaml
```

**Panoptic Segmentation:**
```bash
python visualization/visualize_panoptic.py --config configs/verse/verse_panoptic_R50.yaml
```

**Semantic Segmentation:**
```bash
python visualization/visualize_semantic.py --config configs/verse/verse_semantic_R50.yaml
```

*Note: Running the script without arguments will pick a random image from your test set. You can specify a specific image using `--input path/to/image.png`.* Results are automatically saved to `Mask2Former/results/`.

---

## 💡 Pro-Tips & Troubleshooting

*   **CUDA Compilation Failed?** See the **🛠️ Compiling CUDA Operators** section above. The #1 cause is a version mismatch between `nvcc` and `torch.version.cuda`.
    *   **Conda Users:** Conda paths are now auto-detected! Just ensure you follow the installation command in section 1 to install the required `cuda-toolkit` and `gcc` packages.
*   **Out of Memory (OOM)?** Mask2Former is memory-intensive. If you get a "CUDA out of memory" error:
    1.  Reduce `SOLVER.IMS_PER_BATCH` in your YAML config (e.g., change from 16 to 2 or 1).
    2.  Reduce `INPUT.IMAGE_SIZE` (e.g., from 1024 to 512).
*   **Dataset Not Found?** Registration is now dynamic! Check the `VERSE_ROOT` path inside your YAML config. It must point to the folder containing your processed 2D slices (usually `../dataset_verse_2d/ade20k`).
*   **Bone Windowing:** Our preprocessing script applies a specific bone window (Hounsfield Units). If your visualizations look "washed out" or completely black, verify the windowing settings in `utils/data_utilities.py`.
