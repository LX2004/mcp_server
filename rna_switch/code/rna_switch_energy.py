import RNA

def analyze_rna_switch(switch_seq, trigger_seq, temperature=37):
    """
    使用 ViennaRNA Python API 计算 RNA switch 的能量特征
    """

    results = {}

    # 设置计算参数
    md = RNA.md()
    md.temperature = temperature

    # 1. Switch 自身折叠能量 (MFE)
    fc_self = RNA.fold_compound(switch_seq, md)
    structure, mfe_self = fc_self.mfe()
    results["MFE_self"] = mfe_self
    results["Structure_self"] = structure

    # 2. Trigger + Switch 杂交能量 (Cofold)
    hybrid_seq = switch_seq + '&' + trigger_seq
    structure_hybrid, mfe_hybrid = RNA.cofold(hybrid_seq)
    results["MFE_hybrid"] = mfe_hybrid
    results["Structure_hybrid"] = structure_hybrid

    # 3. ΔΔG_open 粗略估算
    results["DeltaDeltaG_open"] = mfe_self - mfe_hybrid

    return results


# 示例用法
if __name__ == "__main__":
    
    switch_seq = "AUGCAUUGAUGCUACGGAUACGUAGCUAUGCAUUGAUGCUACGGAU"
    trigger_seq = "UACGUACCUAGCUAUGCUA"

    res = analyze_rna_switch(switch_seq, trigger_seq)
    for k, v in res.items():
        print(f"{k}: {v}")
