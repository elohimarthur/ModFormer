# ModFormer

Python scripts for preparing datasets, training a read-level CpG methylation classifier, and running inference on paired HDF5/BAM inputs.

## Repository layout

- `scripts/prepare_bam.py`: prepare BAM inputs and inject movement tags.
- `scripts/prepare_ground_truth.py`: prepare ground-truth information.
- `scripts/generate_dataset.py`: generate training datasets.
- `scripts/train.py`: train with chromosome-level data splits.
- `scripts/predict.py`: run inference and write methylation tags.
- `src/modformer/`: model and dataset implementations.
- `scripts/test_performance.py`: profile inference performance.

## Setup and usage

Prepare a Python environment with PyTorch, torchaudio, PyTorch Lightning, torchmetrics, NumPy, SciPy, scikit-learn, h5py, pysam, and tqdm. BAM preparation also uses minimap2.

Run commands from the repository root. Inspect available options with:

```bash
python -m scripts.train --help
python scripts/predict.py --help
python scripts/prepare_bam.py --help
```

Training expects processed PT shards. Inference requires HDF5/BAM inputs, a model checkpoint, a reference FASTA, and a reference-mapping JSON file. These resources and generated results are not included.

Pass reference paths explicitly when preparing BAM files. The performance profiling script contains machine-specific absolute paths for its reference, mapping, and checkpoint; update those paths before using it on another machine.
