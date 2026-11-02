# TEDI

Implementation of **"TEDI: Target-Review-based Knowledge Distillation for Accurate Rating Prediction"**.

TEDI transfers rating-relevant knowledge from a teacher using target reviews to a student using only historical reviews. At inference time, only the student is used.

## Requirements

Python 3, PyTorch with CUDA support, NumPy, pandas, and tqdm.

```bash
pip install torch numpy pandas tqdm
```

The script uses `cuda:0`; install a PyTorch build compatible with your CUDA environment.

## Getting Started

### 1. Prepare the data

Extract `Data.zip` and set the input paths in `main()` of `TEDI.py`:

- `train_path`, `val_path`, and `test_path`: training, validation, and test splits.
- `encoded_review`, `user_review`, `item_review`, and `map_pair`: precomputed review representations and user–item pair mappings loaded with `np.load`.

### 2. Run training

```bash
mkdir -p saved_model/DM
python3 TEDI.py --dataset DM
```

The script runs teacher training, knowledge distillation, and final rating predictor training in sequence, then reports test MSE, RMSE, and MAE. Checkpoints are saved in `saved_model/<dataset>/`.

`--dataset` sets the output label only; input paths must be configured in the script. For other options, run `python3 TEDI.py --help`.
