from collections import Counter
import numpy as np
import os
import pandas as pd
from Bio import SeqIO

def calculate_overall_kmer_correlation(dataset1, dataset2, k, flag=False):
    # Step 1: Generate a list of all kmers for each dataset
    kmers_dataset1 = [calculate_kmers(seq, k) for seq in dataset1]
    kmers_dataset2 = [calculate_kmers(seq, k) for seq in dataset2]
    
    # Step 2: Flatten the list of kmer lists to get a single list of kmers for each dataset
    flat_kmers_dataset1 = [kmer for sublist in kmers_dataset1 for kmer in sublist]
    flat_kmers_dataset2 = [kmer for sublist in kmers_dataset2 for kmer in sublist]
    
    # Step 3: Get the frequency of each kmer in each dataset
    freq_dataset1 = get_kmer_frequencies(flat_kmers_dataset1)
    freq_dataset2 = get_kmer_frequencies(flat_kmers_dataset2)
    
    # Step 4: Create pandas series for each frequency distribution, ensuring a common index
    s1 = pd.Series(freq_dataset1).fillna(0)
    s2 = pd.Series(freq_dataset2).fillna(0)
    common_index = s1.index.union(s2.index)
    s1 = s1.reindex(common_index, fill_value=0)
    s2 = s2.reindex(common_index, fill_value=0)
    
    # Step 5: Compute the Pearson correlation for the kmers
    correlation = s1.corr(s2)
    
    if flag :
        return s1,s2,correlation
    
    else:
        return correlation
    
# Function to calculate k-mers of a sequence
def calculate_kmers(sequence, k):
    return [sequence[i:i+k] for i in range(len(sequence)-k+1)]


# Function to get the frequency of each k-mer
def get_kmer_frequencies(kmers):
    kmer_counts = Counter(kmers)
    total_kmers = sum(kmer_counts.values())
    # Normalize the counts by the total number of kmers to get frequency
    kmer_freq = {kmer: count / total_kmers for kmer, count in kmer_counts.items()}
    return kmer_freq

'''从生成的fasta文件中获取合成启动子'''
def get_promoter_by_fasta_file(file_name):
    sequences = []

    # 读取fasta文件
    with open(file_name, 'r') as fasta_file:
        # 使用SeqIO.parse函数解析fasta文件
        for record in SeqIO.parse(fasta_file, 'fasta'):
            # 将每个启动子的序列添加到sequences列表中
            sequences.append(str(record.seq))
    
    print(sequences[0])
    print('生成启动子数目为：', len(sequences))

    return sequences

if __name__ == '__main__':


    '''MCP 合成启动子'''
    ss1 = []
    ss2 = []
    corres = []

    nat_sequences = get_promoter_by_fasta_file(file_name='../data/e_coli_by_wx.fasta')
    syn_sequences = get_promoter_by_fasta_file(file_name='output_sequences.fasta')

    for k in range(2,7):

        s1,s2,corre = calculate_overall_kmer_correlation(dataset1=nat_sequences, dataset2=syn_sequences, k=k, flag=True)

        s1 = s1.values
        s2 = s2.values
        # 构造一个 DataFrame
        df = pd.DataFrame({
            'Nature': s1,
            'Synthesis': s2
        })

        # 写入 CSV 文件
        df.to_csv(f'{k}-mer pcc={corre} without opt.csv', index=False)



        print(k,'mer的相关性：', corre)
        print(k,'mer的相关性：', s1)

        ss1.append(s1)
        ss2.append(s2)

        corres.append(corre)

    '''无 MCP 只有提示词的合成启动子'''
    
    ss1 = []
    ss2 = []
    corres = []

    nat_sequences = get_promoter_by_fasta_file(file_name='../data/e_coli_by_wx.fasta')
    syn_sequences = get_promoter_by_fasta_file(file_name='无MCP的提示词和模型.fasta')

    for k in range(2,7):

        s1,s2,corre = calculate_overall_kmer_correlation(dataset1=nat_sequences, dataset2=syn_sequences, k=k, flag=True)

        s1 = s1.values
        s2 = s2.values
        # 构造一个 DataFrame
        df = pd.DataFrame({
            'Nature': s1,
            'Synthesis': s2
        })

        # 写入 CSV 文件
        df.to_csv(f'{k}-mer pcc={corre} 无MCP 有提示词.csv', index=False)


        print(k,'mer的相关性：', corre)
        print(k,'mer的相关性：', s1)

        ss1.append(s1)
        ss2.append(s2)

        corres.append(corre)


    '''无 MCP 有提示词+数据的合成启动子'''
    
    ss1 = []
    ss2 = []
    corres = []

    nat_sequences = get_promoter_by_fasta_file(file_name='../data/e_coli_by_wx.fasta')
    syn_sequences = get_promoter_by_fasta_file(file_name='无MCP的提示词+数据和模型.fasta')

    for k in range(2,7):

        s1,s2,corre = calculate_overall_kmer_correlation(dataset1=nat_sequences, dataset2=syn_sequences, k=k, flag=True)

        s1 = s1.values
        s2 = s2.values
        # 构造一个 DataFrame
        df = pd.DataFrame({
            'Nature': s1,
            'Synthesis': s2
        })

        # 写入 CSV 文件
        df.to_csv(f'{k}-mer pcc={corre} 无MCP 有提示词+数据.csv', index=False)


        print(k,'mer的相关性：', corre)
        print(k,'mer的相关性：', s1)

        ss1.append(s1)
        ss2.append(s2)

        corres.append(corre)
