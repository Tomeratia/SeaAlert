



# SeaAlert - Stage 2 (EDA + Baselines)

SeaAlert is an NLP project for **maritime radio (VHF/SAR) severity classification** into 4 classes:
**Routine, Safety, Urgency, Distress**.

Stage 2 focuses on:

* EDA + data quality checks
* Training **strong but simple baselines**
* Testing robustness against **shortcut learning** from maritime codewords (MAYDAY / PAN PAN / SECURITE)

---

## What’s in Stage 2

### Dataset (current snapshot)

* **300 synthetic VHF/SAR messages**
* **Balanced**: 75 samples per class
* Core fields: `text`, `label`
* Metadata: `scenario_type`, `style`, `vessel`, `location`, `weather`, `pob`
* Codeword fields: `has_codeword`, `codeword`, `actual_codeword`, `text_masked`

### Split

* Train: **240**
* Test: **60**
* **NONE-only** in test (`actual_codeword = NONE`): **31**

---

## Baselines

* **Model A (classical):** TF-IDF (1-2 grams) + Logistic Regression
* **Model B (pretrained):** DistilBERT fine-tuning (`distilbert-base-uncased`) for 4-class classification

---

## Evaluation setup

We report results in 3 settings:

1. **Full** - standard test split
2. **NONE-only** - test samples with `actual_codeword = NONE`
3. **Full (masked input)** - replace codewords with `[CODEWORD]`

---

## Results (Stage 2)

| Model      | Setting             | Accuracy | Macro-F1 |
| ---------- | ------------------- | -------- | -------- |
| BoW+LR     | Full                | 0.9167   | 0.9187   |
| DistilBERT | Full                | 0.9167   | 0.9164   |
| BoW+LR     | NONE-only           | 0.8387   | 0.8185   |
| DistilBERT | NONE-only           | 0.8387   | 0.8393   |
| BoW+LR     | Full (masked input) | 0.7333   | 0.7030   |
| DistilBERT | Full (masked input) | 0.8333   | 0.8321   |

---

## Key findings

* Masking codewords causes a clear drop, **strongest for BoW+LR** → shortcut reliance on codewords.
* **DistilBERT is more robust** under masking and performs best on **NONE-only**.
* Typical BoW+LR errors on NONE-only: **third-party RELAY** messages (Distress/Urgency) predicted as **Routine**.

---

## Repository files (Stage 2)

* `notebooks/SeaAlerts_EDA.ipynb`
  Exploratory Data Analysis and data quality checks for the 300-sample dataset. Includes class balance, field completeness (missingness), codeword statistics, and sanity checks to validate the synthetic generation logic.

* `notebooks/baselines.ipynb`
  End-to-end baseline training and evaluation notebook. Trains:

  * TF-IDF (1-2 grams) + Logistic Regression
  * DistilBERT fine-tuning (`distilbert-base-uncased`)
    Reports metrics under the three evaluation settings: Full, NONE-only, and Masked-input, including confusion matrices and comparison plots.

* `src/make_sample.py`
  Synthetic sample generation module. Defines the structured spec schema (label, scenario, style, vessel/location/weather/POB, codeword policy) and builds LLM prompts to generate grounded VHF/SAR messages. Can export generated samples to CSV for downstream EDA and model training.

---
