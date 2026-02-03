# Experiment Results & Visualizations

This directory contains the outputs of all experiments, evaluations, and data analyses performed in the SeaAlert project. The files are organized into two subdirectories: `csv/` for raw data tables and `visuals/` for plots and qualitative reports.

---

## csv/ (Data Tables)

| File | Description | Source Notebook |
| :--- | :--- | :--- |
| `all_model_results.csv` | **Main Results Table:** Contains performance metrics (Accuracy, Macro F1) for all tested models (BoW vs. RoBERTa) across different settings (Clean vs. ASR, Codeword Masking, etc.). | `04_train_and_evaluate.ipynb` |
| `wer_report.csv` | **ASR Error Analysis:** Breakdown of Word Error Rate (WER) across different noise levels (Low/Med/High) and scenarios. Used to analyze speech-to-text quality. | `00_eda_audio_asr.ipynb` / `03_noise_and_asr.ipynb` |
| `split_indices.csv` | **Data Splits:** Stores the exact indices used for Train (70%), Validation (15%), and Test (15%) sets to ensure reproducibility across all experiments. | `01_generate_synthetic_dataset.ipynb` |
| `results_asr.csv` | **Experiment 3 Results:** Specific F1 scores comparing model robustness on noisy ASR transcripts versus clean text. | `04_train_and_evaluate.ipynb` |
| `results_masking.csv` | **Experiment 1 Results:** Results from the "Codeword Masking" experiment, testing how much models rely on specific keywords like "MAYDAY". | `04_train_and_evaluate.ipynb` |
| `results_trap.csv` | **Experiment 2 Results:** Performance on "Adversarial Traps" (e.g., negations, drills), testing the models' contextual understanding. | `04_train_and_evaluate.ipynb` |
| `trap_set.csv` | **Adversarial Dataset:** The specific set of "trap" examples created to fool keyword-based classifiers. | `04_train_and_evaluate.ipynb` |

---

## visuals/ (Plots & Reports)

### Exploratory Data Analysis (EDA)
| File | Description |
| :--- | :--- |
| `eda_label_distribution.png` | Bar chart showing the balanced distribution of the 4 severity labels (Distress, Urgency, Safety, Routine). |
| `eda_text_length.png` | Distribution of message lengths (word count), showing the variability of the synthetic dataset. |
| `eda_codewords.png` | Analysis of GMDSS codeword usage (MAYDAY, PAN PAN) frequency across different labels. |
| `eda_wordclouds.png` | Word clouds visualizing the most frequent terms for each severity label. |
| `eda_style_scenario.png` | Heatmap or bar chart showing the distribution of communication styles (Formal/Informal) and scenarios. |
| `eda_spectrograms.png` | Visual comparison of clean vs. noisy audio spectrograms (Low/Med/High noise levels). |
| `eda_audio_duration.png` | Histogram of the audio file durations. |
| `eda_wer_analysis.png` | Plot showing how Word Error Rate (WER) increases with noise levels (SNR). |

### Model Performance
| File | Description |
| :--- | :--- |
| `model_comparison_bars.png` | **Key Chart:** Comparison of F1 scores between Baseline (BoW) and Transformer (RoBERTa) models. |
| `performance_degradation.png` | Visualization of performance drop when moving from clean text to noisy ASR inputs. |
| `cm_bow_setting_c.png` | Confusion Matrix for the Bag-of-Words (BoW) baseline model. |
| `cm_tf_setting_c.png` | Confusion Matrix for the RoBERTa transformer model. |
| `summary_bars.png` | Summary visualization of the main 3 experiments (Masking, Traps, ASR). |

### Qualitative Reports
| File | Description |
| :--- | :--- |
| `data_quality_report.txt` | Text summary of dataset statistics and quality checks. |
| `error_analysis.txt` | Detailed analysis of specific misclassification errors made by the models. |
| `demo_asr_roberta_report.json` | Sample JSON output from the inference demo, showing classification and extracted entities (Vessel, Location, etc.). |
