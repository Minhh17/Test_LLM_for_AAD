# -*- coding: utf-8 -*-
import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
from spafe.features.gfcc import gfcc

def extract_gfcc(audio_path, num_ceps=13):
    """
    Trả về:
        gfcc_feat : ndarray shape (frames, num_ceps)
        sr        : sample-rate (để làm trục thời gian)
    """
    sr, y = wav.read(audio_path)           # y có thể là int16 → chuyển sang float
    y = y.astype(float)
    gfcc_feat = gfcc(y, fs=sr, num_ceps=num_ceps)
    return gfcc_feat, sr

def show_gfcc(gfcc_feat, sr, hop_length=512):
    """
    Visualize GFCC dưới dạng heat-map.
    hop_length chỉ để tính trục thời gian (512 mẫu ≈ 32 ms với 16 kHz).
    """
    # (frames, num_ceps)  →  (num_ceps, frames) để hiển thị như spectrogram
    gfcc_disp = gfcc_feat.T

    # Tạo vector thời gian
    frames = gfcc_feat.shape[0]
    times = np.arange(frames) * hop_length / sr

    plt.figure(figsize=(10, 4))
    plt.imshow(gfcc_disp,
               aspect='auto',
               origin='lower',
               interpolation='nearest')
    plt.colorbar(format='%+2.0f dB')
    plt.title("GFCC")
    plt.ylabel("Cepstral Coefficients")
    plt.xlabel("Time (s)")
    plt.xticks(
        ticks=np.linspace(0, frames-1, 6),
        labels=[f"{t:.1f}" for t in np.linspace(0, times[-1], 6)]
    )
    plt.tight_layout()
    plt.show()

# ----- chạy thử -----
gfcc_feat, sr = extract_gfcc("/home/haiminh/Desktop/Data_Predictive_Maintainance/abnormal/abnormal_angle_miss/splited0804_err3_1_0.wav", num_ceps=13)
print("GFCC shape:", gfcc_feat.shape)
show_gfcc(gfcc_feat, sr)

