import re
from Bio.Seq import Seq
from Bio import SeqIO

def find_potential_targets_positive_strand_with_shift(grna, genome_seq, pam="NGG", max_mismatch=3, shift_range=3):
    """
    Search potential off-target sites on the positive strand only, allowing PAM to shift forward
    by 0 to shift_range nt.
    Args:
        grna (str): 20-nt target gRNA sequence
        genome_seq (str): genomic DNA sequence (uppercase string)
        pam (str): PAM pattern (default: "NGG")
        max_mismatch (int): maximum allowed mismatches
        shift_range (int): maximum upstream shift of PAM (default: 3 nt)
    """
    matches = []
    pam_regex = pam.replace("N", "[ACGT]")

    for shift in range(0, shift_range + 1):  # shift = 0,1,2,3
        for i in range(shift, len(genome_seq) - 23 + shift):
            guide_seq = genome_seq[i - shift : i - shift + 20]
            pam_seq   = genome_seq[i - shift + 20 : i - shift + 23]

            if len(guide_seq) != 20 or len(pam_seq) != 3:
                continue

            if re.fullmatch(pam_regex, pam_seq):
                mismatches = sum(nt1 != nt2 for nt1, nt2 in zip(grna, guide_seq))
                if mismatches <= max_mismatch:
                    matches.append({
                        'shift': shift,
                        'start': i - shift,
                        'end': i - shift + 23,
                        'gRNA_match': guide_seq,
                        'PAM': pam_seq,
                        'mismatches': mismatches,
                        'strand': '+'
                    })
    unique_sites = {}
    for s in matches:
        key = (s['start'], s['end'], s['gRNA_match'], s['PAM'], s['strand'])
        if key not in unique_sites or s['shift'] < unique_sites[key]['shift']:
            unique_sites[key] = s
    matches = list(unique_sites.values())

    return matches


def find_potential_targets_negetive_strand_with_shift(grna, genome_seq, pam="NGG", max_mismatch=3, shift_range=3):
    """
    Search potential off-target sites on the negative strand only, allowing PAM to shift forward
    by 0 to shift_range nt.
    Args:
        grna (str): 20-nt target gRNA sequence
        genome_seq (str): genomic DNA sequence (uppercase string)
        pam (str): PAM pattern (default: "NGG")
        max_mismatch (int): maximum allowed mismatches
        shift_range (int): maximum upstream shift of PAM (default: 3 nt)
    """
    matches = []
    pam_regex = pam.replace("N", "[ACGT]")

    for shift in range(0, shift_range + 1):  # shift = 0,1,2,3
        for i in range(shift, len(genome_seq) - 23 + shift):
            guide_seq = genome_seq[i - shift : i - shift + 20]
            pam_seq   = genome_seq[i - shift + 20 : i - shift + 23]

            if len(guide_seq) != 20 or len(pam_seq) != 3:
                continue

            if re.fullmatch(pam_regex, pam_seq):
                mismatches = sum(nt1 != nt2 for nt1, nt2 in zip(grna, guide_seq))
                if mismatches <= max_mismatch:
                    matches.append({
                        'shift': shift,
                        'start': i - shift,
                        'end': i - shift + 23,
                        'gRNA_match': guide_seq,
                        'PAM': pam_seq,
                        'mismatches': mismatches,
                        'strand': '-'
                    })

    unique_sites = {}
    for s in matches:
        key = (s['start'], s['end'], s['gRNA_match'], s['PAM'], s['strand'])
        if key not in unique_sites or s['shift'] < unique_sites[key]['shift']:
            unique_sites[key] = s
    matches = list(unique_sites.values())

    return matches



def get_reverse_complement(dna_sequence):
    """Return reverse complement of the given DNA sequence."""
    return str(Seq(dna_sequence).reverse_complement())



def main_function_off_target(grna, max_mismatch=2, shift_range=3, pam="NGG"):
    # Given a gRNA, search the E. coli genome for potential off-target sites

    # Load genome FASTA
    record = next(SeqIO.parse("../data/ecoli.fasta", "fasta"))
    genome_seq = str(record.seq).upper()

    # Search potential targets on both strands
    sites = find_potential_targets_positive_strand_with_shift(
        grna, genome_seq, pam=pam, max_mismatch=max_mismatch, shift_range=shift_range
    )
    sites += find_potential_targets_negetive_strand_with_shift(
        grna,
        get_reverse_complement(dna_sequence=genome_seq),
        pam=pam,
        max_mismatch=max_mismatch,
        shift_range=shift_range
    )

    return sites

def find_pattern_substrings(seq, length=20):
    """
    Given a DNA sequence, find all 20-nt candidate guides followed by an NGG PAM.
    """
    substrings = []
    for i in range(len(seq) - length - 2):  # -2 because PAM 'NGG' has length 3
        candidate = seq[i:i+length]
        pam = seq[i+length:i+length+3]
        # Check that PAM matches N + GG
        if len(pam) == 3 and pam[1:] == "GG":
            substrings.append(candidate)
    return substrings


if __name__ == "__main__":

    main_function_off_target(grna="GACGTTGACGGTTAGTGTTT", max_mismatch=2, shift_range=3, pam="NGG")
