import os
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import spearmanr
import torch

# -----------------------------------------------
# One-hot encoding and decoding
# -----------------------------------------------

def one_hot_encoding(sequence):
    bases = ['A', 'C', 'G', 'T']
    base_dict = dict(zip(bases, range(4)))
    sequence = sequence.upper()
    length = len(sequence)
    encoded_sequence = np.zeros((4, length), dtype=int)
    for i, base in enumerate(sequence):
        if base in base_dict:
            encoded_sequence[base_dict[base], i] = 1
    return encoded_sequence

def decode_one_hot(one_hot_array):
    base_mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    decoded_sequence = []
    for row in range(one_hot_array.shape[1]):
        max_index = np.argmax(one_hot_array[:, row])
        base = base_mapping[max_index]
        decoded_sequence.append(base)
    return ''.join(decoded_sequence)

# -----------------------------------------------
# k-mer analysis utilities
# -----------------------------------------------
def calculate_kmers(sequence, k):
    return [sequence[i:i+k] for i in range(len(sequence)-k+1)]

def get_kmer_frequencies(kmers):
    kmer_counts = Counter(kmers)
    total_kmers = sum(kmer_counts.values())
    return {kmer: count / total_kmers for kmer, count in kmer_counts.items()}

def calculate_overall_kmer_correlation(dataset1, dataset2, k, flag=False):
    kmers_dataset1 = [calculate_kmers(seq, k) for seq in dataset1]
    kmers_dataset2 = [calculate_kmers(seq, k) for seq in dataset2]
    flat_kmers_dataset1 = [kmer for sublist in kmers_dataset1 for kmer in sublist]
    flat_kmers_dataset2 = [kmer for sublist in kmers_dataset2 for kmer in sublist]
    freq_dataset1 = get_kmer_frequencies(flat_kmers_dataset1)
    freq_dataset2 = get_kmer_frequencies(flat_kmers_dataset2)
    s1 = pd.Series(freq_dataset1).fillna(0)
    s2 = pd.Series(freq_dataset2).fillna(0)
    common_index = s1.index.union(s2.index)
    s1 = s1.reindex(common_index, fill_value=0)
    s2 = s2.reindex(common_index, fill_value=0)
    correlation = s1.corr(s2)
    return (s1, s2, correlation) if flag else correlation

# -----------------------------------------------
# Pearson correlation and regression metrics
# -----------------------------------------------
def loss_pierxun(output, target):
    target_mean = torch.mean(target)
    output_mean = torch.mean(output)
    target_var = torch.std(target)
    output_var = torch.std(output)
    p = torch.mean((output - output_mean) * (target - target_mean))
    if output_var == 0:
        p /= ((output_var + 1e-7) * target_var)
        return p
    p /= (output_var * target_var)
    return p

def evaluate_regression_metrics(output, label):
    if len(output) != len(label):
        raise ValueError("Output and label lengths do not match")
    if np.isnan(output).any() or np.isnan(label).any():
        raise ValueError("NaN values detected in inputs")
    mse = np.mean((output - label) ** 2)
    ss_res = np.sum((label - output) ** 2)
    ss_tot = np.sum((label - np.mean(label)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    spearman_corr = 0 if np.std(label) == 0 or np.std(output) == 0 else spearmanr(label, output)[0]
    return mse, r2, spearman_corr

def compute_correlation_coefficient(output, label):
    target = output
    prediction = label
    if np.isnan(prediction).any() or np.isnan(target).any():
        print("NaN values detected in arrays")
    if np.std(prediction) == 0:
        print('Prediction has zero variance')
        return 0
    if np.std(target) == 0:
        print('Target has zero variance')
        return 0
    mean_target = np.mean(target)
    mean_prediction = np.mean(prediction)
    covariance = np.mean((target - mean_target) * (prediction - mean_prediction))
    std_target = np.std(target)
    std_prediction = np.std(prediction)
    pearson_coefficient = covariance / (std_target * std_prediction)
    return pearson_coefficient

# -----------------------------------------------
# Other utility functions
# -----------------------------------------------
def text_build_vocab():
    dic = [a for a in 'ATCG']
    dic += [a + b for a in 'ATCG' for b in 'ATCG']
    dic += [a + '0' for a in 'ATCG']
    return dic

def Dimer_split_seqs(seq):
    t = text_build_vocab()
    ori_result, dim_result, pos_result = [], [], []
    lens = len(seq)
    for i in range(lens):
        ori_result.append(t.index(seq[i].upper()))
    seq += '0'
    for i in range(lens):
        dim_result.append(t.index(seq[i:i+2].upper()))
    pos_result += [i for i in range(1, lens + 1)]
    seq_r = [ori_result, dim_result, pos_result]
    return seq_r

def write_good_record(dict1, dict2, file_path):
    merged_dict = {**dict1, **dict2}
    if not os.path.isfile(file_path):
        with open(file_path, 'w') as file:
            file.write(f"{merged_dict}\n")
    else:
        with open(file_path, 'a') as file:
            file.write(f"{merged_dict}\n")

def make_fasta_file(sequences, path):
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write("")
        print(f"File {path} does not exist, created a new file.")
    else:
        print(f"File {path} already exists.")
    with open(path, 'w') as file:
        for i, seq in enumerate(sequences, start=1):
            seq = seq.upper()
            file.write(f'>Sequence_{i}\n')
            file.write(f'{seq}\n')
    print(f"File {path} created and sequences written successfully.")

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
