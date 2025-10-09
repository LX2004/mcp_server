import re
from Bio.Seq import Seq
from Bio import SeqIO

def load_fasta_sequence(filepath="../data/ecoli.fasta"):
    """
    从FASTA文件读取DNA序列，返回一个大写字符串
    忽略描述行（以">"开头）
    """
    sequence_lines = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            sequence_lines.append(line.upper())
    return ''.join(sequence_lines)

def get_reverse_complement(dna_sequence):
    """将正义链转换为反义链（reverse complement）"""
    return str(Seq(dna_sequence).reverse_complement())

genome_seq=load_fasta_sequence()
with open("../data/ecoli_positive_sequence.txt", "w") as f:
    f.write(genome_seq)

with open("../data/ecoli_negetive_sequence.txt", "w") as f:
    f.write(get_reverse_complement(genome_seq))