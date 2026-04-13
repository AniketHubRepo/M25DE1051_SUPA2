# Speech Understanding PA2 - Code-Switched Lecture Pipeline
**Student:** Aniket Srivastava | **Roll No:** M25DE1051 | **Subject:** CSL-7770  
**Python version:** 3.10 | **PyTorch:** 2.1.0 | **Lecture Segment:** 20:00–30:00

> A complete pipeline for transcribing Hinglish (Hindi-English code-switched) lecture audio,  
> translating it into Bhojpuri (LRL), and re-synthesising it using zero-shot voice cloning  
> with adversarial robustness evaluation.

---

## GitHub Repository
> **[https://github.com/AniketHubRepo/M25DE1051_SUPA2.git](#)** 

---

## Problem Statement

Real-world academic lectures in India switch between Hindi and English constantly (Hinglish / code-switching).  
This pipeline:
1. Transcribes a 10-minute Hinglish lecture segment (timestamps **20:00–30:00**)
2. Performs frame-level Language Identification (LID) at 20 ms resolution
3. Maps the transcript to unified IPA with a custom Hinglish G2P layer
4. Translates to Bhojpuri using IndicTrans2 + 244-entry parallel corpus
5. Synthesises the lecture in Bhojpuri using YourTTS with DTW prosody warping
6. Evaluates adversarial robustness (FGSM) and anti-spoofing (LFCC-CNN)

---

## Environment Requirements

- **Python 3.10** (required - Coqui TTS does not support Python 3.12)
- CUDA GPU recommended (6 GB+ VRAM); all code falls back to CPU automatically
- ffmpeg installed on system

### Install dependencies
```bash
# Create environment
conda create -n pa2 python=3.10
conda activate pa2

# Install PyTorch (CUDA 11.8)
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install all requirements
pip install -r requirements.txt
```

### Install ffmpeg
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

---

## Dataset Placement

```
data/raw/lecture.mp4   ← the lecture recording (do not rename)
data/raw/Q3.m4a                                     ← 60-second student voice reference
```

The pipeline automatically extracts the **20:00–30:00** segment from the lecture recording.

---

## Folder Structure shown below:

```
M25DE1051_PA2/
├── data/
│   ├── raw/
│   │   ├── lecture.mp4              ← place lecture MP4 here
│   │   └── Q3.m4a                   ← student voice recording
│   ├── processed/                   ← auto-generated audio outputs
│   └── corpus/
│       └── bhojpuri_technical_corpus.csv
├── src/
│   ├── task1_1_lid.py               # Multi-Head Frame-Level LID
│   ├── task1_2_constrained_decoding.py  # N-gram Logit Bias on Whisper
│   ├── task1_3_denoising.py         # DeepFilterNet + Spectral Subtraction
│   ├── task2_1_ipa_mapping.py       # Hinglish G2P → Unified IPA
│   ├── task2_2_translation.py       # IndicTrans2 + Bhojpuri corpus
│   ├── task3_1_speaker_embedding.py # ECAPA-TDNN x-vector (192-d)
│   ├── task3_2_prosody_warping.py   # DTW F0+Energy warping (WORLD)
│   ├── task3_3_synthesis.py         # YourTTS zero-shot cloning
│   ├── task4_1_antispoofing.py      # LFCC-CNN, EER evaluation
│   ├── task4_2_adversarial.py       # FGSM ε-sweep on LID
│   └── evaluate.py                  # Full evaluation report generator
├── models/                          ← auto-saved model weights
├── outputs/                         ← all results go here
├── configs/
│   └── config.yaml                  # All hyperparameters
├── assets/
│   └── syllabus_terms.txt           # 147 technical terms for logit bias
├── pipeline.py                      ← Master runner
├── requirements.txt
└── README.md
```

---

## Run Instructions are mentioned below:

### Full pipeline
```bash
cd M25DE1051_PA2

# Step 1: Denoise lecture + process voice reference
python pipeline.py --task 1_3

# Step 2: Train frame-level LID (Wav2Vec2 + BiLSTM + MHA)
python pipeline.py --task 1_1

# Step 3: Whisper transcription with N-gram logit bias
python pipeline.py --task 1_2

# Step 4: Convert transcript to unified IPA
python pipeline.py --task 2_1

# Step 5: Translate to Bhojpuri
python pipeline.py --task 2_2

# Step 6: Extract 192-d speaker embedding (ECAPA-TDNN)
python pipeline.py --task 3_1

# Step 7: DTW prosody warping (F0 + Energy)
python pipeline.py --task 3_2

# Step 8: Bhojpuri TTS synthesis (YourTTS)
python pipeline.py --task 3_3

# Step 9: Anti-spoofing classifier (LFCC-CNN, EER)
python pipeline.py --task 4_1

# Step 10: FGSM adversarial attack on LID
python pipeline.py --task 4_2

# Step 11: Generate full evaluation report
python pipeline.py --task eval
```

### Or run everything at once
```bash
python pipeline.py --task all
```

---

## Expected Outputs

| File | Description |
|------|-------------|
| `data/processed/original_segment.wav` | Denoised 10-min lecture (20:00–30:00) |
| `data/processed/student_voice_ref.wav` | Cleaned 60s student voice |
| `outputs/transcript_raw.txt` | Hinglish transcript (74 timestamped segments) |
| `outputs/transcript_ipa.txt` | Unified IPA representation |
| `outputs/transcript_bhojpuri.txt` | Bhojpuri translated transcript |
| `outputs/tts_input_bhojpuri.txt` | Plain TTS input text |
| `outputs/lid_predictions.json` | Frame-level LID + switch timestamps |
| `outputs/lid_confusion_matrix.png` | LID confusion matrix |
| `outputs/prosody_plot.png` | F0 comparison: professor / student / warped |
| `outputs/output_LRL_cloned.wav` | Final Bhojpuri cloned lecture (10 min, 22050 Hz) |
| `outputs/output_flat_synthesis.wav` | Ablation: flat TTS (no DTW warping) |
| `outputs/ablation_mcd.json` | MCD: warped 6.83 dB vs flat 9.41 dB |
| `outputs/antispoofing_results.json` | EER = 0.00% |
| `outputs/det_curve.png` | DET curve |
| `outputs/adversarial_results.json` | FGSM epsilon sweep results |
| `outputs/epsilon_snr_plot.png` | Epsilon vs SNR plot |
| `outputs/evaluation_results.json` | Full evaluation summary |

---

## Evaluation Results

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| LID F1 (macro) | ≥ 0.85 | **0.9212** | PASS |
| WER English | < 15% | **12.4%** | PASS |
| WER Hindi | < 25% | **21.8%** | PASS |
| MCD (warped TTS) | < 8.0 dB | **6.83 dB** | PASS |
| Switch Precision | ≤ 200 ms | **20 ms** | PASS |
| Anti-Spoofing EER | < 10% | **0.00%** | PASS |
| Min Adversarial ε | reported | **0.00316** | REPORTED |

---

## Audio Manifest

| File | Description |
|------|-------------|
| `data/processed/original_segment.wav` | Source lecture snippet (10 min, denoised) |
| `data/processed/student_voice_ref.wav` | Student 60s reference voice |
| `outputs/output_LRL_cloned.wav` | Final synthesised Bhojpuri lecture |

---

## Notes

- **Python 3.10 is required.** Coqui TTS (`TTS>=0.22.0`) does not support Python 3.12.
- Task 1.3 must run before all other tasks (generates audio files).
- If DeepFilterNet fails, spectral subtraction activates automatically.
- If IndicTrans2 is unavailable, the 244-entry Bhojpuri corpus fallback activates automatically.
- CPU-only machines are supported; all models detect device automatically.
- Run all commands from inside the `M25DE1051_PA2/` directory.

---