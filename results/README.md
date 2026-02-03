# Experiment Results & Visualizations

This directory contains the outputs of all experiments, evaluations, and data analyses performed in the SeaAlert project. The files are organized into two subdirectories: `csv/` for raw data tables and `visuals/` for plots and qualitative reports.

---

## csv/ (Data Tables)

| File | Description | Source Notebook |
| :--- | :--- | :--- |
| `all_model_results.csv` | **Main Results Table:** Consolidated performance metrics (Accuracy, Macro F1) for all experiments: 1) Clean vs ASR, 2) Codeword Masking, and 3) Adversarial Traps. | `04_train_and_evaluate.ipynb` |


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

