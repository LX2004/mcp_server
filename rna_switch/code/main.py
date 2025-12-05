import os
import time
from datetime import datetime
import argparse
import torch
from utils import *
import script_utils
import os
import cma
import RNA
from rna_switch_energy import analyze_rna_switch
from Bio import SeqIO
from Bio.Seq import Seq


def dna_reverse_complement_to_rna(dna_seq: str) -> str:
    """
    Take a DNA sequence, compute the reverse complement, and convert it to RNA (T->U).
    """
    dna_seq = dna_seq.upper()
    rc_seq = str(Seq(dna_seq).reverse_complement())  # reverse complement
    rna_seq = rc_seq.replace("T", "U")               # convert T to U
    return rna_seq


def construct_toehold_trigger_from_list(merged_list):
    """
    Input a list of merged sequences, each 45 nt: switch(30) + stem1(6) + stem2(9),
    and return the full toehold switch sequence list and corresponding trigger RNAs.

    Args:
    - merged_list: List[str], each element is a 45-nt merged sequence

    Returns:
    - full_sequences: List[str], full toehold switch sequences (RNA)
    - Ttigger_rna: List[str], corresponding trigger RNA sequences
    """
    # fixed scaffold
    loop1 = "AACCAAACACACAAACGCAC"
    loop2 = "AACAGAGGAGA"
    atg = "ATG"
    linker = "AACCTGGCGGCAGCGCAAAAGATGCG"
    post_linker = "TAAAGGAGAA"

    full_sequences = []
    Ttigger_rna = []

    for merged_seq in merged_list:
        if len(merged_seq) != 45:
            raise ValueError(f"序列长度应为45 nt，但发现长度为 {len(merged_seq)}：{merged_seq}")
        
        switch = merged_seq[:30]
        stem1 = merged_seq[30:36]
        stem2 = merged_seq[36:]
        
        full_seq = loop1 + switch + loop2 + stem1 + atg + stem2 + linker + post_linker
        full_seq_rna = full_seq.replace("T", "U")
        full_sequences.append(full_seq_rna)

        Ttigger_rna.append(dna_reverse_complement_to_rna(switch))

    return full_sequences, Ttigger_rna


def process_toehold_structures(switch):
    """
    Given a list of (DNA) toehold segments, construct full switch and trigger,
    compute RNA secondary structure and thermodynamic metrics.
