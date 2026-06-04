# Conda Environments

This project uses two local Conda environments.

## `tgmt`

Use `tgmt` for Mask2Former-based methods:

- baseline Mask2Former;
- focal loss;
- elastic augmentation;
- focal loss + elastic augmentation;
- 2.5D adjacent-slice input;
- multi-window data variant, using the baseline source.

Create it with:

```bash
conda env create -f envs/tgmt.yml
conda activate tgmt
```

Then compile Mask2Former CUDA operators inside each source folder you want to run:

```bash
cd source/Mask2Former-baseline/mask2former/modeling/pixel_decoder/ops
sh make.sh
```

Repeat the same compile step for other self-contained Mask2Former variants if you run them directly.

## `verse_mm`

Use `verse_mm` for external OpenMMLab comparison models:

- DeepLabV3+;
- UPerNet;
- Mask R-CNN;
- QueryInst.

Create it with:

```bash
conda env create -f envs/verse_mm.yml
conda activate verse_mm
```

This repository does not include the external model source trees. Clone the official repositories separately, then install them in editable mode:

```bash
git clone https://github.com/open-mmlab/mmsegmentation.git semantic/mmsegmentation
git clone https://github.com/open-mmlab/mmdetection.git instance/mmdetection

pip install -e semantic/mmsegmentation
pip install -e instance/mmdetection
```

The VerSe-specific configs for these models are in `configs_for_external_comparison/`.

## CUDA Note

The YAML files target PyTorch `2.6.0+cu124`, matching the environment used for the reported experiments. If the reviewer uses a different CUDA driver/toolkit, install the matching PyTorch CUDA wheel and recompile CUDA operators before running inference.
