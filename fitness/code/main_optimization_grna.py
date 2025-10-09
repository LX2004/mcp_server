from prediction_efficiency import Quickly_predict_efficiency
from calculation_off_target import main_function_off_target, find_pattern_substrings
from energy import calculate_binding_energy_from_grna
from web_internet import Quickly_predict_fitness_E_coli
import pandas as pd


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


def opt_grna(seq):

    results = []
    sgrnas = find_pattern_substrings(seq=seq)

    for sgrna in sgrnas:

        sites = main_function_off_target(grna=sgrna, max_mismatch=2, shift_range=3, pam="NGG")
        efficiency = Quickly_predict_efficiency(sgRNA=sgrna)

        num_off_targets = len(sites)
        grna_energy = calculate_binding_energy_from_grna(sgrna)
        fitness = Quickly_predict_fitness_E_coli(sgRNA=sgrna)

        entry = {
            "grna": sgrna,
            "off target count": num_off_targets,
            "efficiency": efficiency.item(),
            'Binding energy (kcal/mol)':grna_energy,
            'fitness': fitness,
        }
        
        results.append(entry)

    df = pd.DataFrame(results)
    return df



res = opt_grna(seq='CTAGCATCGACTAGCTACGATCAGCTACGACTACTACGATCGACTACGATCAGCTACGTACGGCATCGACTGGCATCGACGTGGCTACAGCTGGCATACGACTAGCTACTACGACTATC')
print(res)

# print(jsonify(res))