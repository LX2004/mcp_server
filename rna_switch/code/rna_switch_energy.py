import RNA

def analyze_rna_switch(switch_seq, trigger_seq, temperature=37):
    """
    Compute energy features of an RNA switch using the ViennaRNA Python API.
    """

    results = {}

    # set calculation parameters
    md = RNA.md()
    md.temperature = temperature

    # 1. MFE of switch self-folding
    fc_self = RNA.fold_compound(switch_seq, md)
    structure, mfe_self = fc_self.mfe()
    results["MFE_self"] = mfe_self
    results["Structure_self"] = structure

    # 2. Hybridization energy of trigger + switch (cofold)
    hybrid_seq = switch_seq + '&' + trigger_seq
    structure_hybrid, mfe_hybrid = RNA.cofold(hybrid_seq)
    results["MFE_hybrid"] = mfe_hybrid
    results["Structure_hybrid"] = structure_hybrid

    # 3. Approximate ΔΔG_open
    results["DeltaDeltaG_open"] = mfe_self - mfe_hybrid

    return results


# Example usage
if __name__ == "__main__":
    
    switch_seq = "AUGCAUUGAUGCUACGGAUACGUAGCUAUGCAUUGAUGCUACGGAU"
    trigger_seq = "UACGUACCUAGCUAUGCUA"

    res = analyze_rna_switch(switch_seq, trigger_seq)
    for k, v in res.items():
        print(f"{k}: {v}")
