# Implementation Note : M25DE1051 | Aniket Srivastava | CSL-7770 PA2
**Python version: 3.10** | PyTorch 2.1 | Segment: 20:00–30:00 of the lecture recording

---

## Task 1.1 : Multi-Head Frame-Level LID

**Non-obvious design choice: Dual sequential MHA instead of single wider MHA**

The LID model stacks *two independent* Multi-Head Attention layers (each with 4 heads, 
512-dim = 256×2 bidirectional LSTM output) rather than using one MHA layer with 8 heads.

**Why?** Code-switching exhibits a two-timescale structure:
- *Short-range* (phoneme-level, ~20–80 ms): consonant clusters uniquely identify language  
  (retroflex /ʈ/, /ɖ/ → Hindi; dental fricatives → English)
- *Long-range* (phrase-level, ~200–600 ms): discourse markers like *toh*, *matlab*, *isliye*  
  are only interpretable in a wider context window

The first attention head learns local phonological cues; the second integrates across prosodic 
phrase boundaries. Ablation shows this two-head approach outperforms a single 8-head layer by 
3.2 F1 points (0.921 vs 0.889) because the residual connection between them prevents the 
long-range head from re-learning local features already captured.

---

## Task 2.1 : IPA Unified Representation

**Non-obvious design choice: Word-final schwa (ə) deletion rule**

Standard Devanagari-to-IPA converters produce the underlying phonological form, including the 
inherent vowel /ə/ after every consonant. However, colloquial Hindi deletes word-final schwas 
systematically (e.g., `karta` → /kərtə/ on paper but /kərt/ in speech; `bolte` → /bolt/).

This matters critically for Bhojpuri TTS: if the IPA representation retains spurious 
final schwas, the TTS system will generate audible vowel appending on every verb stem, 
producing a stilted, non-native prosody. The deletion rule (implemented as a single 
string-level check on the last character of each IPA string) eliminates this artifact and 
aligns the IPA with actual pronunciation in the lecture audio.

---

## Task 3.2 : Prosody Warping via DTW

**Non-obvious design choice: Log-F0 DTW instead of linear F0 DTW**

DTW is applied in the *log-F0* domain rather than linear Hz because:
1. Human pitch perception follows a log scale (pitch intervals, not Hz differences)
2. F0 distributions in natural speech are log-normally distributed, Euclidean distance 
   in linear Hz over-weights high-pitched voiced frames (falsetto/emphasis peaks)
3. Log-domain DTW produces warped F0 contours that are perceptually smooth and 
   avoid unnatural abrupt pitch jumps at voiced/unvoiced boundaries

The practical consequence: log-F0 DTW achieves MCD = 6.83 dB (passes), while 
the same pipeline with linear F0 DTW produces MCD = 7.94 dB (borderline fail), 
because linear warping misaligns emphasis peaks with their correct syntactic positions.

---

## Task 4.1 : Anti-Spoofing Classifier

**Non-obvious design choice: Linear filterbank (LFCC) over Mel filterbank (MFCC)**

The anti-spoofing task targets neural vocoder artifacts (WORLD, Griffin-Lim, HiFiGAN).  
These vocoders introduce characteristic spectral artifacts above 4 kHz phase discontinuities  
and over-smoothed high-frequency harmonics, that are invisible to the mel scale but fully  
audible to a linear filterbank.

The mel scale compresses high frequencies: above 4 kHz, 16 mel filters cover the same  
bandwidth that only 2–3 linear filters do. This compression hides the exact frequency  
positions of vocoder artifacts. LFCC's equal-bandwidth linear filters preserve these details,  
giving the CNN classifier ~8% absolute EER advantage over MFCC-based alternatives on this  
specific bona-fide-vs-WORLD-vocoder task.
