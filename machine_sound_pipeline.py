# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import librosa
from scipy.signal import butter, lfilter
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

### 1. Tiền xử lý: lọc tần số ###
def bandpass_filter(y, sr, lowcut=100, highcut=6000, order=5):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y_filtered = lfilter(b, a, y)
    return y_filtered

def extract_mfcc_bandpassed(file_path, n_mfcc=13, lowcut=100, highcut=6000):
    y, sr = librosa.load(file_path, sr=None)
    y = bandpass_filter(y, sr, lowcut, highcut)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # (frames, features)

### 2. Chuẩn bị label và data cho phân loại lỗi ###
label_map = {
    'normal': 0,
    'error_type1': 1,
    'error_type2': 2,
    'undefined': 3
}

X = []
y = []
for label_name, label_id in label_map.items():
    files = glob.glob(f'audio/{label_name}/*.wav')
    for file in files:
        mfcc = extract_mfcc_bandpassed(file)
        mfcc_mean = np.mean(mfcc, axis=0)
        X.append(mfcc_mean)
        y.append(label_id)
X = np.stack(X)
y = np.array(y)
print("Data shape for classifier:", X.shape, y.shape)

### 3. Train model phân loại lỗi ###
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)
print("Classification model trained.")
y_pred = clf.predict(X)
print(classification_report(y, y_pred, target_names=list(label_map.keys())))

### 4. KMeans token hóa + Tiny Transformer anomaly detection ###
# Gom tất cả file train cho LLM anomaly detection
normal_files = glob.glob('audio/normal/*.wav')
all_mfcc = []
for file in normal_files:
    mfcc = extract_mfcc_bandpassed(file)
    all_mfcc.append(mfcc)
all_mfcc = np.vstack(all_mfcc)
n_clusters = 64
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(all_mfcc)

def mfcc_to_tokens(mfcc, kmeans):
    return kmeans.predict(mfcc)

# Gom token
all_train_tokens = []
for file in normal_files:
    mfcc = extract_mfcc_bandpassed(file)
    tokens = mfcc_to_tokens(mfcc, kmeans)
    all_train_tokens.extend(tokens.tolist())

vocab_size = n_clusters
seq_length = 16
inputs = []
labels = []
for i in range(0, len(all_train_tokens) - seq_length):
    inputs.append(all_train_tokens[i:i+seq_length])
    labels.append(all_train_tokens[i+seq_length])
inputs = np.array(inputs)
labels = np.array(labels)
print(f"Total train samples for LLM: {len(inputs)}")

def build_tiny_transformer(seq_length, vocab_size, d_model=64, num_heads=2, num_layers=2):
    inputs = keras.Input(shape=(seq_length,), dtype=tf.int32)
    x = layers.Embedding(vocab_size, d_model)(inputs)
    for _ in range(num_layers):
        x1 = layers.LayerNormalization()(x)
        attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x1, x1)
        x = x + attn_output
        x2 = layers.LayerNormalization()(x)
        ffn_output = layers.Dense(d_model, activation="relu")(x2)
        x = x + ffn_output
    x = layers.Flatten()(x)
    outputs = layers.Dense(vocab_size, activation="softmax")(x)
    return keras.Model(inputs, outputs)

model = build_tiny_transformer(seq_length, vocab_size)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
model.fit(inputs, labels, epochs=10, batch_size=64)

def evaluate_perplexity_on_tokens(tokens, model, seq_length):
    total_log_prob = 0
    total = 0
    for i in range(len(tokens) - seq_length):
        input_seq = np.array(tokens[i:i+seq_length]).reshape(1, -1)
        true_token = tokens[i+seq_length]
        pred = model.predict(input_seq, verbose=0)[0]
        prob = pred[true_token]
        prob = max(prob, 1e-8)
        total_log_prob += -np.log(prob)
        total += 1
    return np.exp(total_log_prob / total) if total > 0 else float('inf')

def severity_level(score):
    if score < 50:
        return "Normal"
    elif score < 500:
        return "Warning"
    elif score < 2000:
        return "Alert"
    else:
        return "Critical"

### 5. Inference: Đánh giá từng file ###
all_test_files = []
for label_name in label_map.keys():
    all_test_files += glob.glob(f'audio/{label_name}/*.wav')

print(f"{'File':35s} {'Predicted Type':15s} {'Score':>10s} {'Severity':>10s}")
for file in all_test_files:
    mfcc = extract_mfcc_bandpassed(file)
    mfcc_mean = np.mean(mfcc, axis=0)
    # Phân loại lỗi
    pred_label = clf.predict([mfcc_mean])[0]
    label_name = [k for k,v in label_map.items() if v == pred_label][0]
    # Anomaly score (perplexity)
    tokens = mfcc_to_tokens(mfcc, kmeans)
    if len(tokens) < seq_length + 1:
        sev = "(short file)"
        score = 0
    else:
        score = evaluate_perplexity_on_tokens(tokens, model, seq_length)
        sev = severity_level(score)
    print(f"{os.path.basename(file):35s} {label_name:15s} {score:10.2f} {sev:>10s}")

