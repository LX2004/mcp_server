import re
from Bio.Seq import Seq
from Bio import SeqIO

def load_fasta_sequence(filepath="../data/ecoli.fasta"):
    """
    Read DNA sequence from a FASTA file and return it as an uppercase string.
    Lines starting with '>' (description lines) are ignored.
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
    """Return the reverse complement of a DNA sequence."""
    return str(Seq(dna_sequence).reverse_complement())

genome_seq = load_fasta_sequence()
with open("../data/ecoli_positive_sequence.txt", "w") as f:
    f.write(genome_seq)

with open("../data/ecoli_negetive_sequence.txt", "w") as f:
    f.write(get_reverse_complement(genome_seq))
