**EEG Confusion Classification with Multi-Band 1D-CNN (PyTorch)**

Window single-sensor EEG into 10-s segments (5-s stride), z-score per window, stack raw + band-power features (delta→gamma) as
9 channels, and train a compact 1D-CNN. Evaluate with subject-aware GroupKFold to prevent data leakage.

**Repo contents**

eeg-brainwave-classification.ipynb — end-to-end notebook (segmentation → model → CV → reports)

**Dataset**

Participants: 10 college students

Stimuli: 20 short online-course videos (~2 min each): 10 expected to be non-confusing (e.g., algebra, geometry) and 10 expected to be confusing (e.g., quantum mechanics, stem cells)

Recording: Single-channel wireless MindSet over the frontal area (forehead electrode; ears as ground/reference)

Sampling: One value every 0.5 s (~2 Hz). Higher-frequency band powers are reported as the mean within each 0.5-s slice

Epoch selection: Middle 1 minute of each video (first/last 30 s removed)

Scale: 10 subjects × 10 videos ⇒ 100 trials (~120 time rows per trial at 2 Hz)

Files used:

EEG_data.csv — time-series features and labels

demographic.csv — basic subject info

**Task**

Binary classification: confused vs. not confused. Labels are trial-level (per video); the code predicts at the window level and can aggregate to trial level.

**Pipeline used:**

Segmentation:

      frequency = 2.0 (samples/sec)
    
      window_secs = 10, step_secs = 5 ⇒ T = 20 samples per window, 50% overlap
    
      Per-window metadata tracked: (subject, video, start_time)

Channels (multi-band input):

      ["Raw","Delta","Theta","Alpha1","Alpha2","Beta1","Beta2","Gamma1","Gamma2"] ⇒ (N, C=9, T)

Normalization:

      Per-window, per-channel z-score across time (leak-safe)

Model (PyTorch):
  
      Compact 1D-CNN: Conv1d → BatchNorm1d → ReLU → MaxPool → Conv1d → BN → ReLU → AdaptiveAvgPool → Dropout → Linear
    
      Class-weighted CrossEntropy to address imbalance; Adam (+ weight decay)

Evaluation:

      GroupKFold by SubjectID (5 folds) to avoid subject leakage
    
      Window-level metrics (accuracy, F1) and confusion matrices
