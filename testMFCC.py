import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Hàm trích xuất MFCC từ audio
def extract_mfcc(audio_path, n_mfcc=13, visualize=True):
    # Load file audio (ý nghĩa: audio số hóa thành tín hiệu)
    y, sr = librosa.load(audio_path, sr=None) # sr=None: giữ sampling rate gốc

    print(f"Loaded {audio_path}, sample rate = {sr}, length = {len(y)} samples")

    # Trích xuất MFCC (ý nghĩa: chuyển tín hiệu thành ma trận đặc trưng)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    print(f"MFCC shape = {mfcc.shape}")

    # Hiển thị MFCC để trực quan hóa (nếu visualize=True)
    if visualize:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfcc, x_axis='time', sr=sr)
        plt.colorbar()
        plt.title('MFCC')
        plt.tight_layout()
        plt.show()

    return mfcc.T  # trả về dạng (frames, n_mfcc)

# Test ngay với audio của bạn
#mfcc_feature = extract_mfcc('/home/haiminh/Desktop/Data_Predictive_Maintainance/normal/splited0803_s1_0.wav', n_mfcc=15)
mfcc_feature = extract_mfcc("/home/haiminh/Desktop/Data_Predictive_Maintainance/abnormal/abnormal_angle_miss/splited0804_err3_1_0.wav", n_mfcc=15)

