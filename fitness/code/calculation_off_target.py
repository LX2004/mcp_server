import re
from Bio.Seq import Seq
from Bio import SeqIO

def find_potential_targets_positive_strand_with_shift(grna, genome_seq, pam="NGG", max_mismatch=3, shift_range=3):
    """
    只在正义链（+）上搜索潜在脱靶位点，考虑 PAM 前 shift（0~shift_range nt）错位
    grna: 长度为20的目标gRNA序列
    genome_seq: DNA序列（大写字符串）
    pam: PAM模式（默认为NGG）
    max_mismatch: 允许最大错配碱基数
    shift_range: 最大向前偏移碱基数（默认最多3nt）
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
    只在正义链（+）上搜索潜在脱靶位点，考虑 PAM 前 shift（0~shift_range nt）错位
    grna: 长度为20的目标gRNA序列
    genome_seq: DNA序列（大写字符串）
    pam: PAM模式（默认为NGG）
    max_mismatch: 允许最大错配碱基数
    shift_range: 最大向前偏移碱基数（默认最多3nt）
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
    """将正义链转换为反义链（reverse complement）"""
    return str(Seq(dna_sequence).reverse_complement())



def main_function_off_target(grna, max_mismatch=2, shift_range=3, pam="NGG"):
    # 输入grna，然后在大肠杆菌全基因组上寻找潜在的脱靶位点

    # 加载FASTA文件
    record = next(SeqIO.parse("../data/ecoli.fasta", "fasta"))
    genome_seq = str(record.seq).upper()

    # 查找潜在靶点（只在正义链、允许2错配，PAM前最多错位3nt）
    sites = find_potential_targets_positive_strand_with_shift(grna, genome_seq, pam=pam, max_mismatch=max_mismatch, shift_range=shift_range)
    sites += find_potential_targets_negetive_strand_with_shift(grna, get_reverse_complement(dna_sequence=genome_seq), pam=pam, max_mismatch=max_mismatch, shift_range=shift_range)

    # 打印结果统计
    # print(f"共找到 {len(sites)} 个正义链潜在靶点（含错位）")
    # for s in sites[:5]:
    #     print(s)

    return sites

def find_pattern_substrings(seq, length=20):

    """
    根据给定的靶向序列seq，查找grna。
    """
    
    substrings = []
    for i in range(len(seq) - length - 2):  # -2 是因为 'NGG' 长度为3
        candidate = seq[i:i+length]
        pam = seq[i+length:i+length+3]
        # 检查 PAM 是否为 N + GG
        if len(pam) == 3 and pam[1:] == "GG":
            substrings.append(candidate)
    return substrings


if __name__ == "__main__":


    main_function_off_target(grna="GACGTTGACGGTTAGTGTTT", max_mismatch=2, shift_range=3, pam="NGG")
