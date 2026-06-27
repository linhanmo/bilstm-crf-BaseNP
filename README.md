# BaseNP

BaseNP is a Chinese base noun phrase sequence labeling project built on CTB 8.0.

The project currently supports:

- Automatic preprocessing from `LDC2013T21.tgz` to `CTB/`
- Automatic conversion from CTB parse trees to `data/basenp/`
- One-command training for `HMM`, `CRF`, `BiLSTM`, and `BiLSTM-CRF`
- Automatic comparison with a `jieba` baseline
- Unified experiment outputs under `outputs/`

## Environment

Recommended Python version:

```bash
python --version
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

Place the CTB 8.0 archive in the project root:

```text
LDC2013T21.tgz
```

The archive is only used locally for preprocessing and is ignored by Git.

## Quick Start

Run the full pipeline:

```bash
python training.py
```

The pipeline will automatically:

1. Check whether `data/basenp/` already exists
2. If needed, run `preprocess.py` to prepare `CTB/`
3. Convert `CTB/par/*.noempty.txt` into BaseNP TSV files
4. Train all models
5. Run comparison experiments

## Main Commands

Run the full pipeline and force rebuilding BaseNP data:

```bash
python training.py --rebuild-data
```

Skip data preparation and only train/evaluate:

```bash
python training.py --skip-data-prepare
```

Skip comparison after training:

```bash
python training.py --skip-compare
```

Train a single model manually:

```bash
python train_hmm.py
python train_crf.py
python train_bilstm.py
python train_bilstm_crf.py
```

Run comparison manually:

```bash
python compare_models.py
```

## Data Format

`data/basenp/*.tsv` uses sentence-separated TSV format:

```text
上海    B-NP
浦东    I-NP
开发    I-NP
建设    E-NP
同步    O
```

Rules:

- The first column is the token
- The last column is the BaseNP tag
- Sentences are separated by blank lines

## BaseNP Extraction

The current CTB-to-BaseNP conversion follows a practical "maximal acceptable NP" strategy:

- Prefer larger noun phrases over overly fragmented nested NP chunks
- Filter out phrases containing clause-like or prepositional structures
- Filter out phrases with `DEC/DEG/DEV/DER`
- Reduce noisy single-token chunks such as pure time or determiner phrases

Implementation entry:

- `training.py`

## Outputs

All experiment artifacts are stored under:

```text
outputs/
```

Typical layout:

```text
outputs/
  hmm/<run_name>/
  crf/<run_name>/
  bilstm/<run_name>/
  bilstm_crf/<run_name>/
  compare/<run_name>/
  pipeline/<run_name>/
```

## Project Structure

```text
BaseNP/
  preprocess.py
  training.py
  compare_models.py
  config.py
  util.py
  train_hmm.py
  train_crf.py
  train_bilstm.py
  train_bilstm_crf.py
  pseudocode.md
  requirements.txt
```

## Notes

- All default paths in the project use relative paths
- `CTB/`, `data/`, `outputs/`, and `LDC2013T21.tgz` are local resources and should not be committed
- If you update the BaseNP extraction rules, rerun:

```bash
python training.py --rebuild-data
```
