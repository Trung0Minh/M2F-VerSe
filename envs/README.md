# Conda Environments

This project uses two local Conda environments.

## `verse_detectron2`

Use `verse_detectron2` for Mask2Former-based methods:

- baseline Mask2Former;
- focal loss;
- elastic augmentation;
- focal loss + elastic augmentation;
- 2.5D adjacent-slice input;
- multi-window data variant, using the baseline source.

Create it with:

```bash
conda env create -f envs/verse_detectron2.yml
conda activate verse_detectron2
```

The YAML installs and pins the CUDA 12.4 compiler/runtime headers and CUDA math library headers from `nvidia/label/cuda-12.4.1`. This is required even when PyTorch itself can see the GPU, because Detectron2 and Mask2Former compile custom CUDA/C++ operators during setup. Keep these CUDA 12.4 pins together; allowing newer CUDA subpackages can move the Thrust headers or omit headers such as `cusparse.h`, breaking extension builds.
It also pins `libstdcxx-ng` from conda-forge so Detectron2's compiled `_C` extension can find the required `GLIBCXX` symbols at import time.

Then install Detectron2 after PyTorch is already available in the environment:

```bash
python -m pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/detectron2.git@b599f139756bd3646a26a909caf86a1a159e53a7"
```

This separate step is intentional. Installing Detectron2 inside the Conda YAML can fail because pip builds it in isolation before `torch` is visible.

Then compile Mask2Former CUDA operators inside each source folder you want to run:

```bash
cd source/Mask2Former-baseline/mask2former/modeling/pixel_decoder/ops
sh make.sh
```

Repeat the same compile step for other self-contained Mask2Former variants if you run them directly.

## `verse_openmmlab`

Use `verse_openmmlab` for external OpenMMLab comparison models:

- DeepLabV3+;
- UPerNet;
- Mask R-CNN;
- QueryInst.
- MaskDINO.

Create it with:

```bash
conda env create -f envs/verse_openmmlab.yml
conda activate verse_openmmlab
```

Install the OpenMMLab CUDA extension package after PyTorch is already available:

```bash
python -m pip install "setuptools==69.5.1"
python -m pip install --no-build-isolation "mmcv==2.2.0"
```

This separate step avoids pip building `mmcv` before PyTorch and CUDA headers are visible. The reported experiments used `mmcv==2.2.0`, `mmengine==0.10.7`, `mmdet==3.3.0`, and `mmsegmentation==1.2.2`.

This repository does not include the external model source trees. Clone the matching official release tags separately, then install them in editable mode:

```bash
git clone --branch v1.2.2 https://github.com/open-mmlab/mmsegmentation.git semantic/mmsegmentation
git clone --branch v3.3.0 https://github.com/open-mmlab/mmdetection.git instance/mmdetection
python - <<'PY'
from pathlib import Path
patches = [
    (Path('semantic/mmsegmentation/mmseg/__init__.py'), "MMCV_MAX = '2.2.0'", "MMCV_MAX = '2.3.0'"),
    (Path('instance/mmdetection/mmdet/__init__.py'), "mmcv_maximum_version = '2.2.0'", "mmcv_maximum_version = '2.3.0'"),
]
for path, old, new in patches:
    text = path.read_text()
    if old in text:
        path.write_text(text.replace(old, new))
PY

python -m pip install --no-build-isolation --no-deps -e semantic/mmsegmentation
python -m pip install --no-build-isolation --no-deps -e instance/mmdetection
```

The two-line compatibility patch matches the source trees used for the reported experiments. The release tags otherwise reject `mmcv==2.2.0` because their version guards use `<2.2.0`.
The `--no-deps` flag is intentional because the required runtime packages are already pinned in `envs/verse_openmmlab.yml`; letting pip resolve dependencies again can pull incompatible packages or require internet access during editable installation.

The VerSe-specific configs for these models are in `configs_for_external_comparison/`.

MaskDINO is also treated as an external source tree. Use the official MaskDINO repository and the VerSe configs in `configs_for_external_comparison/semantic/maskdino_R50_semantic_verse.yaml` and `configs_for_external_comparison/instance/maskdino_R50_instance_verse.yaml`.

## CUDA Note

The YAML files target PyTorch `2.6.0+cu124`, matching the environment used for the reported experiments. If the reviewer uses a different CUDA driver/toolkit, install the matching PyTorch CUDA wheel and recompile CUDA operators before running inference.
Run commands from an activated Conda shell (`conda activate ...`) instead of wrapping long CUDA jobs with `conda run`; on some NVIDIA driver setups, `conda run` can report inconsistent NVML/CUDA availability for script files even when the activated environment works normally.
