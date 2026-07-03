# SeaAlert - Project Page

This repository contains the project page for the paper:

**"SeaAlert: Robust Severity Classification and LLM-Based Information Extraction for Noisy Maritime Distress Communications"**

Published in IEEE Access, 2026. DOI: [10.1109/ACCESS.2026.3709004](https://ieeexplore.ieee.org/document/11592329)

## Authors

- **Tomer Atia** - School of Computer Science, HIT - Holon Institute of Technology, Israel
- Yehudit Aperstein - Intelligent Systems, Afeka Academic College of Engineering, Tel Aviv, Israel
- Alexander Apartsin - School of Computer Science, HIT - Holon Institute of Technology, Israel

## About

SeaAlert is a controlled experimental framework for evaluating robust analysis of maritime distress communications. It combines synthetic data generation with VHF noise simulation, ASR transcription, transformer-based severity classification, and LLM-based structured information extraction to assess robustness under realistic operational conditions.

### Key Contributions

- A **synthetic data generation pipeline** using GPT-4o-mini producing 1,872 labeled maritime distress messages across 4 severity classes and 12 scenario types
- A **noisy audio simulation pipeline** using Coqui TTS and VHF channel noise injection with Whisper ASR transcription
- A comparative evaluation of a **rule-based GMDSS keyword spotter, Bag-of-Words, and RoBERTa** classifiers under clean, noisy, codeword-masked, and adversarial conditions
- A comparison of **Regex vs. GPT-4o-mini** for structured information extraction under ASR noise

## Project Page

The live project page is available at: [https://tomeratia.github.io/SeaAlert](https://tomeratia.github.io/SeaAlert)

## Repository Structure

```
SeaAlert/
├── index.html              # Main project page
├── .nojekyll               # Disables Jekyll processing on GitHub Pages
├── README.md               # This file
└── static/
    ├── css/                # Stylesheets (Bulma, carousel, slider, FontAwesome)
    ├── js/                 # Scripts (jQuery, carousel, slider, FontAwesome)
    └── images/             # Figures and pipeline diagram
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{Atia2026SeaAlert,
  title={SeaAlert: Robust Severity Classification and LLM-Based Information
         Extraction for Noisy Maritime Distress Communications},
  author={Atia, Tomer and Aperstein, Yehudit and Apartsin, Alexander},
  journal={IEEE Access},
  year={2026},
  doi={10.1109/ACCESS.2026.3709004},
  url={https://ieeexplore.ieee.org/document/11592329}
}
```
