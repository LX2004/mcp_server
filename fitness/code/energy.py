import RNA

# Nearest-neighbor model parameters
NN_PARAMS = {
    'rAA/dTT': [-7.8, -21.3], 'rAU/dTA': [-8.3, -23.9], 'rAG/dTC': [-9.1, -23.5], 'rAC/dTG': [-5.9, -12.3],
    'rUA/dAT': [-7.8, -22.6], 'rUU/dAA': [-10.5, -29.5], 'rUG/dAC': [-10.4, -28.4], 'rUC/dAG': [-8.2, -21.5],
    'rGA/dCT': [-9.0, -26.1], 'rGU/dCA': [-10.4, -28.4], 'rGG/dCC': [-12.8, -31.9], 'rGC/dCG': [-16.3, -47.1],
    'rCA/dGT': [-9.3, -25.5], 'rCU/dGA': [-7.0, -18.1], 'rCG/dGC': [-10.1, -24.4], 'rCC/dGG': [-9.6, -23.2],
    'init': [1.9, -3.9]
}

def get_reverse_complement(seq):
    """DNA reverse complement"""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'U': 'A'}
    return "".join(complement.get(base, 'N') for base in reversed(seq))

def calculate_delta_gu(grna_seq):
    """Free energy of gRNA self-folding"""
    (_ss, mfe) = RNA.fold(grna_seq)
    return mfe

def calculate_delta_gh(grna_seq, target_dna_seq, temp_celsius=37.0):
    """RNA–DNA hybridization free energy"""
    temp_kelvin = temp_celsius + 273.15
    total_dh, total_ds = NN_PARAMS['init']
    if len(grna_seq) != len(target_dna_seq):
        raise ValueError("gRNA and DNA sequences must have the same length")
    target_dna_seq_3_5 = target_dna_seq[::-1]
    for i in range(len(grna_seq) - 1):
        rna_dimer = grna_seq[i:i+2]
        dna_dimer_3_5 = target_dna_seq_3_5[i:i+2]
        nn_key = f'r{rna_dimer}/d{dna_dimer_3_5}'
        if nn_key in NN_PARAMS:
            dh, ds = NN_PARAMS[nn_key]
            total_dh += dh
            total_ds += ds
    return total_dh - temp_kelvin * (total_ds / 1000.0)

def calculate_binding_energy_from_grna(grna_seq, pam_seq="NGG", temp_celsius=37.0):
    """
    Input:
        grna_seq: gRNA (20 nt, DNA or RNA)
    Automatically constructs:
        target DNA = reverse complement of gRNA + PAM
    Output:
        total binding free energy (kcal/mol)
    """
    grna_seq = grna_seq.upper().replace('T', 'U')
    protospacer_dna = get_reverse_complement(grna_seq.replace('U', 'T'))
    target_dna_seq = protospacer_dna + pam_seq

    delta_gu = calculate_delta_gu(grna_seq)
    delta_gh = calculate_delta_gh(grna_seq, protospacer_dna, temp_celsius)
    delta_go = 5.9
    is_canonical_pam = (len(pam_seq) == 3 and pam_seq[1] == 'G' and pam_seq[2] == 'G')
    delta_pam = 1.0 if is_canonical_pam else 0.2
    total_binding_energy = delta_pam * delta_gh - delta_gu - delta_go

    return total_binding_energy
    

# Example
if __name__ == "__main__":
    
    grna_list = [
        "CGGCGGTGTTGGCCAGGTTC",
        "GTTCGACAAATTTTGTAAGG",
        "GGGTGCGGGCGATCACTTCC",
        "CTGGTTTTTCAGTTTACGCA",
        "TGTTGGCCTTTAACGAACTC",
        "CCCAGCCACAGCCAGGTAAC",
        "GCAGACAGCAACCAGTCAGA",
        "AGCGGGTATCGACCTGCTGC",
        "CCGCTTTATTGACGAGTTAA",
        "CAATAAAGCGGTAGGTTTCC",
        "CGCCCTGAACGAACTGCCGA",
        "AGATCGCATTCTGGCGCTGG",
        "TACAAACTGGGCAGCAGGGC",
        "CGGTAATGGTAACTTCTTGC",
        "CAGTTTGTAAAAGAAGCTAA",
        "CCTTAGCTTCTTTTACAAAC"
    ]
    for grna in grna_list:
        result = calculate_binding_energy_from_grna(grna)
        print(grna, "##", result)
