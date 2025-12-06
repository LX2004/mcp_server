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
    Take a DNA sequence, compute the reverse complement, then convert to RNA (T -> U).
    """
    dna_seq = dna_seq.upper()
    rc_seq = str(Seq(dna_seq).reverse_complement())
    rna_seq = rc_seq.replace("T", "U")
    return rna_seq


def construct_toehold_trigger_from_list(merged_list):
    """
    Input:
        merged_list: list of merged sequences, each 45 nt:
                     switch(30) + stem1(6) + stem2(9)

    Return:
        full_sequences: list of full toehold switch RNA sequences
        Ttigger_rna: list of trigger RNA sequences (reverse complement of switch)
    """
    # Fixed parts of the toehold design
    loop1 = "AACCAAACACACAAACGCAC"
    loop2 = "AACAGAGGAGA"
    atg = "ATG"
    linker = "AACCTGGCGGCAGCGCAAAAGATGCG"
    post_linker = "TAAAGGAGAA"

    full_sequences = []
    Ttigger_rna = []

    for merged_seq in merged_list:
        if len(merged_seq) != 45:
            raise ValueError(f"Sequence length must be 45 nt, but got {len(merged_seq)}: {merged_seq}")
        
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
    For a list of merged switch sequences:
      1. Build full toehold RNA and trigger RNA.
      2. Compute RNA secondary structure and thermodynamic features.

    Returns:
        toehold_switch_sequence: list of full switch RNAs
        Trigger_rnas: list of trigger RNAs
        DeltaDeltaG_opens: list of ΔΔG_open values
        MFE_selfs: list of MFE_self values
    """

    toehold_switch_sequence, Trigger_rnas = construct_toehold_trigger_from_list(merged_list=switch)

    MFE_selfs = []
    DeltaDeltaG_opens = []
    records = []

    for seq, trigger in zip(toehold_switch_sequence, Trigger_rnas):
        structure, mfe = RNA.fold(seq)
        res = analyze_rna_switch(switch_seq=seq, trigger_seq=trigger)

        DeltaDeltaG_opens.append(round(res["DeltaDeltaG_open"], 4))
        MFE_selfs.append(round(res["MFE_self"], 4))

        records.append(
            {
                "Sequence": seq,
                "Trigger rna": trigger,
                "Structure": structure,
                "MFE_self (kcal/mol)": round(res["MFE_self"], 2),
                # "MFE (kcal/mol)": round(mfe, 2),
                "MFE_hybrid (kcal/mol)": round(res["MFE_hybrid"], 2),
                "DeltaDeltaG_open (kcal/mol)": round(res["DeltaDeltaG_open"], 2),
            }
        )

    return toehold_switch_sequence, Trigger_rnas, DeltaDeltaG_opens, MFE_selfs


def construct_toehold_from_list(merged_list):
    """
    Input:
        merged_list: list of merged sequences, each 45 nt:
                     switch(30) + stem1(6) + stem2(9)

    Return:
        full_sequences: list of full toehold switch DNA sequences (not converted to RNA)
    """
    loop1 = "AACCAAACACACAAACGCAC"
    loop2 = "AACAGAGGAGA"
    atg = "ATG"
    linker = "AACCTGGCGGCAGCGCAAAAGATGCG"
    post_linker = "TAAAGGAGAA"

    full_sequences = []

    for merged_seq in merged_list:
        if len(merged_seq) != 45:
            raise ValueError(f"Sequence length must be 45 nt, but got {len(merged_seq)}: {merged_seq}")
        
        switch = merged_seq[:30]
        stem1 = merged_seq[30:36]
        stem2 = merged_seq[36:]
        
        full_seq = loop1 + switch + loop2 + stem1 + atg + stem2 + linker + post_linker
        full_sequences.append(full_seq)

    return full_sequences


def build_structure_array(seq: str) -> np.ndarray:
    """
    Build an N×N structure matrix based on base-pairing potential.
    Each cell stores the number of hydrogen bonds (0, 2, or 3).

    Pairing rules:
        A–U / U–A: 2
        G–C / C–G: 3
        G–U / U–G: 2

    Input:
        seq (str): input RNA or DNA sequence (T will be treated as U)

    Return:
        np.ndarray of shape (N, N)
    """
    seq = seq.upper().replace('T', 'U')
    N = len(seq)
    struct_array = np.zeros((N, N), dtype=int)

    pair_map = {
        ('A', 'U'): 2, ('U', 'A'): 2,
        ('G', 'C'): 3, ('C', 'G'): 3,
        ('G', 'U'): 2, ('U', 'G'): 2
    }

    for i in range(N):
        for j in range(N):
            pair = (seq[i], seq[j])
            if pair in pair_map:
                struct_array[i, j] = pair_map[pair]

    return struct_array


def encode_structure_list(seqList):
    """
    Given a list of merged toehold segments:
      1. Build full mRNA sequences.
      2. Encode them as:
         - Dimer-based token indices (X_seq)
         - Structure matrices (structures)

    Returns:
        X_seq: np.ndarray, encoded sequence indices
        structures: np.ndarray, N×N structural matrices
    """

    # First convert merged segments to full mRNA sequences
    seqList = construct_toehold_from_list(merged_list=seqList)

    structures = np.array([build_structure_array(seq=mRNA) for mRNA in seqList])
    X_seq = np.array([Dimer_split_seqs(sequence) for sequence in seqList])

    return X_seq, structures


def compute_scaler(model_output):
    """
    Simple passthrough: detach tensor and convert to NumPy.
    """
    model_output = model_output.detach().cpu().numpy()
    return model_output


def create_argparser(promoters_number):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    defaults = dict(
        num_images=promoters_number,
        device=device,
        schedule_low=1e-4,
        schedule_high=0.02,
        out_init_conv_padding=1
    )
    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--save_dir", type=str)
    script_utils.add_dict_to_argparser(parser, defaults)

    return parser


def main_function(promoters_number, opt=False, kind='radio'):
    """
    Main entry:
      1. Generate candidate sequences via diffusion model.
      2. Predict ON and OFF expression using trained predictors.
      3. Optionally run CMA-ES optimization on latent tensor.

    Returns:
        If opt=False:
            ons, offs, sequences
        If opt=True:
            all_ons, all_offs, all_sequences from optimization
    """

    args = create_argparser(promoters_number=promoters_number).parse_args()

    model_path = '../model/generated_switch_model.pth'
    diffusion = script_utils.get_diffusion_from_args(args).to(device)
    diffusion.load_state_dict(torch.load(model_path, weights_only=False))

    prediction_on = torch.load('../model/prediction_on.pth', weights_only=False).to(device)
    prediction_off = torch.load('../model/prediction_off.pth', weights_only=False).to(device)

    for name, param in prediction_on.named_parameters():
        print(f"{name}: {param.data}")
        break

    for name, param in prediction_off.named_parameters():
        print(f"{name}: {param.data}")
        break

    sequences = []

    samples = diffusion.sample(args.num_images, device)
    samples = samples.squeeze(dim=1)
    samples = samples.to('cpu').detach().numpy()

    for j in range(samples.shape[0]):
        decoded_sequence = decode_one_hot(samples[j])
        sequences.append("A" + decoded_sequence)

    # Predict expression levels
    Dimer, struc = encode_structure_list(sequences)
    Dimer = torch.tensor(Dimer).to(device)
    struc = torch.tensor(struc).to(device)

    print(Dimer.shape)
    print(struc.shape)

    ons = compute_scaler(prediction_on(Dimer, struc))
    offs = compute_scaler(prediction_off(Dimer, struc))

    ons = np.array(ons).squeeze()
    offs = np.array(offs).squeeze()

    print('offs = ', offs)

    if opt:
        all_sequences, all_ons, all_offs, all_radios = optimize_tensor_input_with_cmaes(
            diffusion_model=diffusion,
            predictor_on=prediction_on,
            predictor_off=prediction_off,
            number=promoters_number,
            sigma=0.5,
            max_iter=30,
            popsize=promoters_number,
            kind=kind
        )
        return all_ons, all_offs, all_sequences

    else:
        return ons.tolist(), offs.tolist(), sequences.tolist()


def blackbox_objective_z_tensor(
    z_flat_np,
    diffusion_model,
    predictor_on,
    predictor_off,
    shape=(512, 1, 4, 44),
    kind='difference'
):
    """
    Black-box objective for CMA-ES over latent tensor:
      1. Decode latent tensor into sequences via diffusion model.
      2. Evaluate ON/OFF predictions and RNA switch thermodynamics.
      3. Compute objective based on 'radio' (on/off) or 'difference' (on - off).

    Returns:
        negative mean score (for minimization),
        sequences, ons, offs, radios, toehold_switch_sequence,
        Trigger_rnas, DeltaDeltaG_opens, MFE_selfs
    """

    sequences = []
    z_tensor = torch.tensor(z_flat_np, dtype=torch.float32).reshape(shape).to(device)

    samples = diffusion_model.sample_opt(x=z_tensor, batch_size=shape[0], device=device)
    samples = samples.squeeze(dim=1)
    samples = samples.to('cpu').detach().numpy()

    for j in range(samples.shape[0]):
        decoded_sequence = decode_one_hot(samples[j])
        sequences.append("A" + decoded_sequence)

    # Energy calculation
    toehold_switch_sequence, Trigger_rnas, DeltaDeltaG_opens, MFE_selfs = process_toehold_structures(switch=sequences)

    # ON/OFF prediction
    Dimer, struc = encode_structure_list(sequences)
    Dimer = torch.tensor(Dimer).to(device)
    struc = torch.tensor(struc).to(device)

    print(Dimer.shape)
    print(struc.shape)

    ons = compute_scaler(predictor_on(Dimer, struc))
    offs = compute_scaler(predictor_off(Dimer, struc))

    ons = np.asarray(ons).squeeze()
    offs = np.asarray(offs).squeeze()

    if kind == 'radio':
        # Prevent division by zero
        offs_safe = np.where(offs == 0, 1e-8, offs)
        radios = ons / offs_safe
    elif kind == 'difference':
        radios = ons - offs
    else:
        raise ValueError("kind must be 'radio' or 'difference'")

    mean_radio = np.nanmean(radios)

    return (
        -mean_radio,
        sequences,
        ons,
        offs,
        radios,
        toehold_switch_sequence,
        Trigger_rnas,
        DeltaDeltaG_opens,
        MFE_selfs,
    )


def tensor_to_promoter(output_tensor):
    """
    Decode a model output tensor (one-hot-like) into a list of promoter sequences.
    """
    samples = output_tensor.squeeze(dim=1)
    samples = samples.to('cpu').detach().numpy()
    sequences = []

    for i in range(samples.shape[0]):
        decoded_sequence = decode_one_hot(samples[i])
        sequences.append(decoded_sequence)

    return sequences


def prediction_strength(prediction_model, promoter):
    """
    Compute the average promoter strength from a prediction model.

    Inputs:
        prediction_model: trained model, maps encoded sequences → raw score
        promoter: list[str], promoter sequences

    Output:
        avg_strength: float, mean strength after exponent transform
    """
    features = [np.array(Dimer_split_seqs(seq)) for seq in promoter]
    features = np.array(features)

    encoded_sequences = torch.tensor(features, dtype=torch.float32).to(device)

    raw_scores = prediction_model(encoded_sequences)

    min_strength = -8.6382
    max_strength = 12.5883
    raw_scores = raw_scores.squeeze().cpu().numpy()

    transformed_scores = 2 ** (raw_scores * (max_strength - min_strength) + min_strength)

    avg_strength = transformed_scores.mean()
    print('avg_strength = ', avg_strength)

    return avg_strength


def optimize_tensor_input_with_cmaes(
    diffusion_model,
    predictor_on,
    predictor_off,
    number=512,
    sigma=0.5,
    max_iter=5,
    popsize=512,
    kind='difference'
):
    """
    Use CMA-ES to optimize a tensor input (e.g., soft one-hot latent)
    to maximize ON/OFF objective through the diffusion model + predictors.

    Returns:
        all_sequences, all_ons, all_offs, all_radios
    """
    shape = (number, 1, 4, 44)
    x = torch.randn(number, 1, 4, 44, device=device)
    x_flat = x.flatten().cpu()
    x = np.array(x_flat)

    es = cma.CMAEvolutionStrategy(x, sigma, {'popsize': popsize})

    best_score = -float('inf')

    all_sequences = []
    all_ons = []
    all_offs = []
    all_radios = []

    for gen in range(max_iter):
        solutions = es.ask()
        scores = []

        for s in solutions:
            (
                score,
                sequences,
                ons,
                offs,
                radios,
                toehold_switch_sequence,
                Trigger_rnas,
                DeltaDeltaG_opens,
                MFE_selfs,
            ) = blackbox_objective_z_tensor(
                s,
                diffusion_model,
                predictor_on,
                predictor_off,
                shape=shape,
                kind=kind,
            )

            df = pd.DataFrame(
                {
                    'on': ons,
                    'off': offs,
                    'radio': radios,
                    'sequence': sequences,
                    'toehold_switch_sequence': toehold_switch_sequence,
                    'Trigger_rnas': Trigger_rnas,
                    'DeltaDeltaG_opens': DeltaDeltaG_opens,
                    'MFE_selfs': MFE_selfs,
                }
            )

            today = datetime.now().strftime("%m%d")
            result_dir = f"../result-{kind}-{today}/"
            os.makedirs(result_dir, exist_ok=True)

            df.to_csv(os.path.join(result_dir, f"output_opt_iter={gen}_score={score}.csv"), index=False)

            all_sequences += sequences
            all_offs += offs.tolist()
            all_ons += ons.tolist()
            all_radios += radios.tolist()

            scores.append(score)

            if -score > best_score:
                best_score = -score

        es.tell(solutions, scores)
        print(f"[Gen {gen}] best average score = {best_score:.4f}")

    return all_sequences, all_ons, all_offs, all_radios


def divide_lists(ons: list, offs: list) -> list:
    """
    Element-wise division of two lists (ons / offs) with simple 0-handling.
    """
    if len(ons) != len(offs):
        raise ValueError("Input lists must have the same length")
    
    result = []
    for on, off in zip(ons, offs):
        if off == 0:
            result.append(float('inf'))
        else:
            result.append(on / off)
    
    return result


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.cuda.set_device(0)


if __name__ == '__main__':

    opt = True
    promoter_number = 32
    ons = []
    sequences = []
    offs = []

    kind = 'radio'

    on, off, sequence = main_function(promoters_number=promoter_number, opt=opt, kind=kind)
    # on, off, sequence = main_function(promoters_number=promoter_number, opt=opt, kind='difference')

    ons += on
    offs += off
    sequences += sequence

    df = pd.DataFrame(
        {
            'on': on,
            'off': off,
            'radio': np.array(on, dtype=float) - np.array(off, dtype=float),
            'sequence': sequences,
        }
    )

    today = datetime.now().strftime("%m%d")
    result_dir = f"../result-{kind}-{today}/"

    if opt:
        df_sorted = df.sort_values(by='radio', ascending=False).head(promoter_number)
        df_sorted.to_csv(result_dir + 'opt_output.csv', index=False)
    else:
        df_sorted = df.sort_values(by='radio', ascending=False).head(promoter_number * 3)
        df_sorted.to_csv(result_dir + 'output.csv', index=False)
