# -*- coding: utf-8 -*-
import librosa
import numpy as np
from sklearn.cluster import KMeans
from nltk.util import ngrams
from collections import Counter
import glob, os

# 1. List files
normal_files = glob.glob('audio/normal/*.wav')
abnormal_files = glob.glob('audio/abnormal/*.wav')
test_files = normal_files + abnormal_files

def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T

all_mfcc = []
for file in normal_files:
    mfcc = extract_mfcc(file)
    all_mfcc.append(mfcc)
all_mfcc = np.vstack(all_mfcc)
n_clusters = 64
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(all_mfcc)

def mfcc_to_tokens(mfcc, kmeans):
    return kmeans.predict(mfcc)

all_train_tokens = []
for file in normal_files:
    mfcc = extract_mfcc(file)
    tokens = mfcc_to_tokens(mfcc, kmeans)
    all_train_tokens.extend(tokens.tolist())

def train_ngram_model(tokens, n=3):
    return Counter(ngrams(tokens, n))

def calc_perplexity(model, tokens, n=3):
    ngram_seq = list(ngrams(tokens, n))
    total = len(ngram_seq)
    log_prob = 0
    total_count = sum(model.values())
    for ng in ngram_seq:
        prob = model[ng] / total_count if model[ng] > 0 else 1e-8
        log_prob += -np.log(prob)
    return np.exp(log_prob / total) if total > 0 else float('inf')

ngram_n = 3
model = train_ngram_model(all_train_tokens, n=ngram_n)

def evaluate_file(file, kmeans, model, ngram_n=3):
    mfcc = extract_mfcc(file)
    tokens = mfcc_to_tokens(mfcc, kmeans)
    perplexity = calc_perplexity(model, tokens, n=ngram_n)
    return perplexity

print(f"{'File':30s} {'Type':12s} {'Perplexity':>10s}")
for file in test_files:
    label = 'Normal' if 'normal' in file else 'Abnormal'
    score = evaluate_file(file, kmeans, model, ngram_n)
    print(f"{os.path.basename(file):30s} {label:12s} {score:10.2f}")

