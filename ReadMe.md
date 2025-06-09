# Vietnamese Tone Classification — README

## Project Overview

This repository implements a deep-learning pipeline for classifying the six Vietnamese tones (ngang, huyền, sắc, hỏi, ngã, nặng) from isolated syllable recordings. We compare three approaches:

1. **Custom CNN** on 64×225 log-mel spectrograms
2. **Wav2Vec2** transfer learning (feature extractor + lightweight head)
3. **ToneNet** (pretrained Mandarin CNN) fine-tuned with discriminative layer-freezing

We generate balanced **synthetic** data via Google Cloud TTS, augment with a small real-speech “drill” set, and evaluate on an unseen Forvo set of native speakers.

---

## Repository Structure

```
├── packaged_data/  
│   ├── subset_package_small/           # 1,800 synth
│   │   ├── X.npy  
│   │   ├── y.npy  
│   │   ├── label_names.json  
│   │   └── metadata_subset.csv  
│   ├── drills_package/                 # 1,557 pronunciation-drills  
│   └── forvo_syllables/                # 52 unseen test syllables  
│       ├── X.npy  
│       ├── y.npy  
│       ├── label_names.json  
│       └── metadata_subset.csv  
│  
├── notebooks/  
│   └── tone_classification.ipynb       # main Colab notebook  
│  
├── scripts_utilities/                  # preprocessing scripts  
│  
├── experiment_results/                 # trained model weights & results  
│  
├── models_and_results/ 				# trained model weights & results  
│
├── project_report.pdf					# report of the project in pdf format  
│
├── project_presentation.pdf			# presentation of the project in pdf format  
│ 
└── README.md                           # this file  
```

---

## Data Availability

Some preprocessed `.npy` packages are too large for GitHub. You can clone/download what’s on GitHub under `packaged_data/`, then **request access** to the remainder here:

> [https://drive.google.com/drive/folders/1RuRA1A\_prVmLAHVnAY22xL2MHFAiDTUl?usp=sharing](https://drive.google.com/drive/folders/1RuRA1A_prVmLAHVnAY22xL2MHFAiDTUl?usp=sharing)

Place each folder (e.g. `subset_package_small`, `drills_package`, etc.) under:

```
/content/drive/MyDrive/UPF_Deep_Learning_2025/Project/
```

so that your data root matches the notebook’s expectations.

---

## Paths in Notebook

```python
local_data_root      = "/content/drive/MyDrive/UPF_Deep_Learning_2025/Project/"
synth_small_pkg_dir  = local_data_root + "subset_package_small"
synth_large_pkg_dir  = local_data_root + "subset_package_large"
drill_pkg            = local_data_root + "drills_package"
combined_pkg_small   = local_data_root + "combined_package_small"
combined_pkg_large   = local_data_root + "combined_pkg_large"
tone_net_models      = local_data_root + "ToneNet_Models"
forvo_pkg            = local_data_root + "forvo_syllables"
```

---

## Installation & Dependencies

We recommend running in **Google Colab** (free GPU) or any environment with:

* Python 3.8+
* `torch`, `torchvision`
* `torchaudio` (for wav loading)
* `onnx`, `onnx2pytorch`
* `transformers` (for wav2vec2)
* `librosa`, `soundfile`
* `opencv-python`, `matplotlib`, `pandas`, `scikit-learn`

Install with:

```bash
pip install torch torchaudio torchvision \
            onnx onnx2pytorch transformers \
            librosa soundfile opencv-python \
            matplotlib pandas scikit-learn
```

---

## Quick Start

1. **Mount Google Drive** in Colab:

   ```python
   from google.colab import drive
   drive.mount('/content/drive', force_remount=True)
   ```

2. **Ensure data folders** are under your Drive root (see “Paths in Notebook”).

3. **Open** `notebooks/tone_classification.ipynb`, run all cells.

4. **Monitor** training logs, view plots of loss/accuracy, and inspect final test/Forvo reports.

---

## Contact

If you have any issues or need data access, please open an issue or contact me through GitHub or email.
