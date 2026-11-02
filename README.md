# TEDI

Implementation of **"TEDI: Target-Review-based Knowledge Distillation for Accurate Rating Prediction"**.

TEDI transfers rating-relevant knowledge from a teacher that uses target reviews to a student that uses only historical reviews. A target review is written by the target user for the target item. It is available during training, but unavailable when predicting a rating for a new user–item interaction.

## Repository Contents

| File | Description |
| --- | --- |
| `3step_PFD.py` | Teacher training, knowledge distillation, and final rating prediction. |
| `Data.zip` | Data archive to extract before configuring input paths. |

## Requirements

- Python 3
- PyTorch with CUDA support
- NumPy, pandas, and tqdm
- A CUDA-capable GPU: the script currently selects `cuda:0` explicitly.

Install a CUDA-compatible PyTorch build for your environment, then install the remaining dependencies:

```bash
python3 -m pip install numpy pandas tqdm
```

The repository does not specify tested package versions. The training script loads precomputed review embeddings; it does not encode raw review text.

## Getting Started

### 1. Clone the repository and extract the data

```bash
git clone https://github.com/Bigdasgit/TEDI.git
cd TEDI
unzip Data.zip -d Data
```

Inspect the extracted files and use their actual locations in the configuration below. Paths are resolved relative to the directory from which you run the script.

### 2. Configure input paths

In `main()` in `3step_PFD.py`, replace the three empty split paths and the four empty paths passed to `np.load`.

The following filenames are examples, not guaranteed names inside the archive:

```python
train_path = 'Data/train.json'
val_path = 'Data/val.json'
test_path = 'Data/test.json'

encoded_review = np.load('Data/encoded_review.npy', allow_pickle=True)
item_review = np.load('Data/item_review.npy', allow_pickle=True)
user_review = np.load('Data/user_review.npy', allow_pickle=True)
map_pair = np.load('Data/map_pair.npy', allow_pickle=True)
```

**Setting `--dataset` does not select or load data files.** It only sets the dataset label used in logs and checkpoint paths. Configure all seven paths directly in the script.

### 3. Check the input format

Each split is a JSON object mapping a user ID to a list of `[rating, item_id]` pairs. For example:

```json
{
  "0": [[5, 2], [3, 7]],
  "1": [[4, 2]]
}
```

| Array | Expected shape | Meaning |
| --- | --- | --- |
| `encoded_review` | `(N, dims)` | Target-review embeddings. The default `dims` is 768. |
| `map_pair` | `(N, 2)` | Corresponding `[user_id, item_id]` for each target-review embedding. |
| `user_review` | `(num_users, 1024)` | Historical-review representations indexed by user ID. |
| `item_review` | `(num_items, 1024)` | Historical-review representations indexed by item ID. |

Use nonnegative integer user and item IDs consistently across splits and arrays. The code derives embedding-table sizes from the maximum IDs in the training split, so validation and test IDs must fall within those ranges.

Keep `map_pair` and `encoded_review` aligned row by row, with one row per unique user–item pair. The current script accesses target reviews for training, validation, and test pairs when evaluating the teacher and extracting teacher embeddings, so the mapping must cover all three splits. The final student rating predictor itself uses only historical-review representations.

The historical-review input width is fixed at 1024 in the model. Changing `--dims` changes the expected target-review width only.

### 4. Create the checkpoint directory

The script does not create its output directory automatically. For the default dataset label:

```bash
mkdir -p saved_model/DM
```

If you change `--dataset`, create `saved_model/<dataset>` for that label.

### 5. Run training and evaluation

After configuring the paths:

```bash
python3 3step_PFD.py --dataset DM
```

Example with explicit hyperparameters:

```bash
python3 3step_PFD.py --dataset DM --batch 128 --lr 0.0005 --alpha 0.57
```

To view all command-line options:

```bash
python3 3step_PFD.py --help
```

## Training Pipeline

One execution runs the following stages in order:

1. **Train the teacher (`TR_predictor`).** Combine target-review embeddings with user and item historical-review representations to predict ratings, minimizing rating MSE.
2. **Distill the teacher's knowledge (`TR_hidden`).** Extract teacher hidden embeddings, then train a student using only historical-review representations to match them. The loss combines embedding MSE and cosine embedding loss: `alpha * MSE + (1 - alpha) * cosine_loss`.
3. **Train the final rating predictor (`Predictor`).** Transfer and freeze the distilled representation layers, then train the rating prediction layers and user/item biases.

The script selects checkpoints using validation rating MSE in stages 1 and 3, and validation embedding MSE in stage 2. Each stage uses early stopping. Console messages retain the original labels `Training 1`, `Training 3`, and `Training 4`.

## Command-Line Options

| Option | Default | Description |
| --- | --- | --- |
| `--dataset` | `DM` | Dataset label for logs and checkpoint directories; does not configure input paths. |
| `--batch` | `128` | Batch size. |
| `--dims` | `768` | Target-review embedding dimension. |
| `--lr` | `0.0005` | Learning rate for Adam. |
| `--emb_hidden` | `768` | Hidden dimension for user/item representation layers. |
| `--hidden` | `320` | Knowledge embedding dimension and rating predictor hidden dimension. |
| `--early_stop` | `30` | Early-stopping patience setting. |
| `--epochs` | `10000` | Maximum epochs per training stage. |
| `--alpha` | `0.57` | Weight on embedding MSE in the distillation loss. |

PyTorch and NumPy seeds are set to 10 in the script.

## Outputs and Evaluation

Checkpoints are saved under `saved_model/<dataset>/` with prefixes `3step_step1_PFD_v2`, `3step_step2_PFD_v2`, and `3step_step3_PFD_v2`. Filenames also include the learning rate, hidden dimension, and selected epoch index.

At the end of a run, the console reports:

- `Step1 RMSE / MSE / MAE`: teacher results, using target reviews.
- `Test RMSE / MSE / MAE`: final student rating prediction results.
- `Test tr / cos`: embedding MSE and cosine embedding loss for the distilled student.

Lower values are better. Rating predictions are clipped to [1, 5] during evaluation. The current evaluator averages per-batch metrics equally, including per-batch RMSE; these can differ from metrics computed over all examples at once.

## Troubleshooting

- **File not found:** check all seven paths in `main()` and run from the repository root.
- **Checkpoint directory does not exist:** create `saved_model/<dataset>/` before training.
- **CUDA unavailable:** install a CUDA-compatible PyTorch build and check GPU availability. CPU execution requires changing the hard-coded device in the script.
- **Shape mismatch:** verify 1024-dimensional historical-review representations and that `--dims` matches target-review embeddings.
- **Index or target-review lookup errors:** check ID ranges, unique pair mappings, and row alignment between `map_pair` and `encoded_review`.
