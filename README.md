# SeaAlert - Stage 2 (EDA + Baselines)

SeaAlert is an NLP project for **maritime radio (VHF/SAR) severity classification** into 4 classes:
**Routine, Safety, Urgency, Distress**.

Stage 2 focuses on:
- Basic EDA + data quality checks
- Training **strong but simple baselines**
- Testing robustness against **shortcut learning** from maritime codewords (MAYDAY / PAN PAN / SECURITE)

---

## What’s in Stage 2

### Dataset (current snapshot)
- **300 synthetic VHF/SAR messages**
- **Balanced**: 75 samples per class (4 labels)
- Core fields: `text`, `label`
- Metadata (examples): `scenario_type`, `style`, `vessel`, `location`, `weather`, `pob`
- Codeword-related fields:
  - `has_codeword`, `codeword`, `actual_codeword`
  - `text_masked` (codewords replaced with `[CODEWORD]`)

### Data split
- Train: **240**
- Test: **60**
- **NONE-only** subset in test (no explicit codeword): **31**

---

## Baselines

### Model A (classical)
- TF-IDF (1-2 grams) + Logistic Regression

### Model B (pretrained)
- DistilBERT fine-tuning (`distilbert-base-uncased`) with a 4-class classification head

---

## Evaluation setup

We report results in 3 settings:
1. **Full test** - standard test split
2. **NONE-only test** - only samples with `actual_codeword = NONE`
3. **Masked input** - replace codewords (MAYDAY / PAN PAN / SECURITE) with `[CODEWORD]` to reduce shortcut signals

---

## Results (Stage 2)

### Summary table (Accuracy, Macro-F1)

| Model | Setting | Accuracy | Macro-F1 |
|------|---------|----------|----------|
| BoW+LR | Full test | 0.9167 | 0.9187 |
| BoW+LR | NONE-only | 0.8387 | 0.8185 |
| BoW+LR | Full (masked input) | 0.7333 | 0.7030 |
| DistilBERT | Full test | 0.9333 | 0.9330 |
| DistilBERT | NONE-only | 0.8710 | 0.8810 |
| DistilBERT | Full (masked input) | 0.8333 | 0.8330 |

Notes:
- DistilBERT masked evaluation is reported on **Full test** (NONE-only masked can be added later if needed).

---

## Key findings

- **Masking codewords causes a clear performance drop**, especially for **BoW+LR**.
  - This suggests **shortcut learning** (reliance on explicit protocol keywords).
- **DistilBERT is more robust under masking** and performs best on the **NONE-only** setting.
- Error pattern (BoW+LR on NONE-only): many failures are **third-party RELAY** messages (Distress/Urgency) predicted as **Routine**.

---

## Repo structure (recommended)


