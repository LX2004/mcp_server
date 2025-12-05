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

    Args:
        switch (List[str]): list of merged sequences (45 nt)

    Returns:
        toehold_switch_sequence (List[str]): full toehold switch RNA sequences
        Trigger_rnas (List[str]): trigger RNA sequences
        DeltaDeltaG_opens (List[float]): ΔΔG_open values
        MFE_selfs (List[float]): self-folding MFEs
    """

    toehold_switch_sequence, Trigger_rnas = construct_toehold_trigger_from_list(merged_list=switch)

    # store structural and energy information
    MFE_selfs = []
    DeltaDeltaG_opens = []
    records = []

    for seq, trigger in zip(toehold_switch_sequence, Trigger_rnas):

        structure, mfe = RNA.fold(seq)
        res = analyze_rna_switch(switch_seq=seq, trigger_seq=trigger)
        DeltaDeltaG_opens.append(round(res["DeltaDeltaG_open"], 4))
        MFE_selfs.append(round(res["MFE_self"], 4))

        records.append({

            "Sequence": seq,
            "Trigger rna": trigger,
            "Structure": structure,

            "MFE_self (kcal/mol)": round(res["MFE_self"], 2),
            # "MFE (kcal/mol)": round(mfe, 2),
            "MFE_hybrid (kcal/mol)": round(res["MFE_hybrid"], 2),
            "DeltaDeltaG_open (kcal/mol)": round(res["DeltaDeltaG_open"], 2),
            
        })

    # df = pd.DataFrame(records)
    # df_sorted = df.sort_values(by="DeltaDeltaG_open (kcal/mol)", ascending=True)

    return toehold_switch_sequence, Trigger_rnas, DeltaDeltaG_opens, MFE_selfs
    # df_sorted.to_csv(output_csv, index=False)


def construct_toehold_from_list(merged_list):
    """
    Input a list of merged sequences, each 45 nt: switch(30) + stem1(6) + stem2(9),
    and return the full toehold switch DNA sequence list.

    Args:
    - merged_list: List[str], each element is a 45-nt merged sequence

    Returns:
    - List[str], each is a full toehold switch sequence (DNA)
    """
    # fixed scaffold
    loop1 = "AACCAAACACACAAACGCAC"
    loop2 = "AACAGAGGAGA"
    atg = "ATG"
    linker = "AACCTGGCGGCAGCGCAAAAGATGCG"
    post_linker = "TAAAGGAGAA"

    full_sequences = []

    for merged_seq in merged_list:
        if len(merged_seq) != 45:
            raise ValueError(f"序列长度应为45 nt，但发现长度为 {len(merged_seq)}：{merged_seq}")
        
        switch = merged_seq[:30]
        stem1 = merged_seq[30:36]
        stem2 = merged_seq[36:]
        
        full_seq = loop1 + switch + loop2 + stem1 + atg + stem2 + linker + post_linker
        # full_seq_rna = full_seq.replace("T", "U")
        full_sequences.append(full_seq)

    return full_sequences


def build_structure_array(seq: str) -> np.ndarray:
    """
    Build an N×N structural array for an RNA sequence, where each position
    encodes the base-pairing potential (in terms of hydrogen bond count).

    Args:
        seq (str): input RNA sequence (e.g. 'AUGCGAU...')

    Returns:
        np.ndarray: N×N structural array, encoded as 0, 2, 3 according to rules
    """
    seq = seq.upper().replace('T', 'U')  # also support DNA input
    N = len(seq)
    struct_array = np.zeros((N, N), dtype=int)

    # valid base pairs and corresponding hydrogen bond counts
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
    Construct full mRNA sequences from toehold segments and encode both
    sequence and structure for prediction models.
    """

    # first build full mRNA sequences, then encode as input for the prediction model
    seqList = construct_toehold_from_list(merged_list=seqList)

    structures = np.array([build_structure_array(seq=mRNA) for mRNA in seqList])
    X_seq = np.array([Dimer_split_seqs(sequence) for sequence in seqList])

    return X_seq, structures


def compute_scaler(model_output):

    model_output = model_output.detach().cpu().numpy()

    return model_output


def create_argparser(promoters_number):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    defaults = dict(num_images=promoters_number, device=device, schedule_low=1e-4,
    schedule_high=0.02,out_init_conv_padding = 1)
    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--save_dir", type=str)
    script_utils.add_dict_to_argparser(parser, defaults)

    return parser


def main_function(promoters_number, opt=False):

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

    # print('strat to generate sequences')
    samples = diffusion.sample(args.num_images, device)
    # print('end to generate sequences')

    # print('samples.shape = ', samples.shape)
    samples = samples.squeeze(dim=1)
    # print('samples.shape = ', samples.shape)

    samples = samples.to('cpu').detach().numpy()

    for j in range(samples.shape[0]):

        decoded_sequence = decode_one_hot(samples[j])
        sequences.append("A" + decoded_sequence)

    """
    Start predicting expression levels
    """

    Dimer, struc = encode_structure_list(sequences)
    Dimer = torch.tensor(Dimer).to(device)
    struc = torch.tensor(struc).to(device)

    print(Dimer.shape)
    print(struc.shape)

    ons = compute_scaler(prediction_on(Dimer, struc))  # predictions for ON state
    offs = compute_scaler(prediction_off(Dimer, struc))  # predictions for OFF state

    ons = np.array(ons)
    ons = ons.squeeze()

    offs = np.array(offs)
    offs = offs.squeeze()

    print('offs = ', offs)
    

    if opt:
        all_sequences, all_ons, all_offs, all_radios = optimize_tensor_input_with_cmaes(
            diffusion_model=diffusion,
            predictor_on=prediction_on,
            predictor_off=prediction_off,
            number=promoters_number, 
            sigma=0.5,
            max_iter=30,
            popsize=promoters_number
        )
        
        return all_ons, all_offs, all_sequences

    else:
         
        return ons, offs, sequences


def blackbox_objective_z_tensor(z_flat_np, diffusion_model, predictor_on, predictor_off, shape=(512, 1, 4, 44)):

    sequences = []
    z_tensor = torch.tensor(z_flat_np, dtype=torch.float32).reshape(shape).to(device)

    # print('strat to generate sequences')
    samples = diffusion_model.sample_opt(x=z_tensor,batch_size=shape[0], device=device)
    # print('end to generate sequences')

    # print('samples.shape = ', samples.shape)
    samples = samples.squeeze(dim=1)
    # print('samples.shape = ', samples.shape)

    samples = samples.to('cpu').detach().numpy()

    for j in range(samples.shape[0]):

        decoded_sequence = decode_one_hot(samples[j])
        sequences.append("A" + decoded_sequence)

    """
    Compute RNA thermodynamic energy features
    """
    toehold_switch_sequence, Trigger_rnas, DeltaDeltaG_opens, MFE_selfs = process_toehold_structures(switch=sequences)

    """
    Predict expression levels
    """

    Dimer, struc = encode_structure_list(sequences)
    Dimer = torch.tensor(Dimer).to(device)
    struc = torch.tensor(struc).to(device)

    print(Dimer.shape)
    print(struc.shape)

    ons = compute_scaler(predictor_on(Dimer, struc))  # ON predictions
    offs = compute_scaler(predictor_off(Dimer, struc))  # OFF predictions

    ons = np.array(ons)
    ons = ons.squeeze()
    # ons = ons.tolist()

    offs = np.array(offs)
    offs = offs.squeeze()
    # offs = offs.tolist()

    radios = ons - offs
    mean_radio = np.nanmean(radios)  # ignore NaN when computing mean

    # still a minimization problem (CMA-ES minimizes, so return negative mean_radio)
    return -mean_radio, sequences, ons, offs, radios, toehold_switch_sequence, Trigger_rnas, DeltaDeltaG_opens, MFE_selfs


def tensor_to_promoter(output_tensor):
    
    samples = output_tensor.squeeze(dim=1)
    samples = samples.to('cpu').detach().numpy()
    sequences = []
    
    for i in range(samples.shape[0]):

        decoded_sequence = decode_one_hot(samples[i])
        sequences.append(decoded_sequence)
                    
    return sequences


def prediction_strength(prediction_model, promoter):
    """
    Args:
        prediction_model: trained prediction model (input: encoded sequence → output: raw score)
        promoter: list of promoter sequences (str)

    Returns:
        avg_strength: average expression strength after exponential transform (float)
    """
    # 1. sequence feature encoding
    features = [np.array(Dimer_split_seqs(seq)) for seq in promoter]
    features = np.array(features)  # shape: (N, F)

    # 2. to Tensor
    encoded_sequences = torch.tensor(features, dtype=torch.float32).to(device)

    # 3. model prediction of raw scores
    raw_scores = prediction_model(encoded_sequences)  # shape: (N,)
    
    # 4. normalization and exponential transform
    min_strength = -8.6382
    max_strength = 12.5883
    raw_scores = raw_scores.squeeze().cpu().numpy()  # ensure numpy array

    transformed_scores = 2 ** (raw_scores * (max_strength - min_strength) + min_strength)

    # 5. average across promoters
    avg_strength = transformed_scores.mean()
    print('avg_strength = ',avg_strength)

    return avg_strength


def optimize_tensor_input_with_cmaes(diffusion_model, predictor_on, predictor_off, number=512, 
                                     sigma=0.5, max_iter=5, popsize=512):
    """
    Use CMA-ES to optimize a tensor input (e.g. soft one-hot representation)
    to maximize the predicted expression strength.

    shape: shape of the optimized tensor, e.g. (64, 4, 44)
    """
    shape=(number, 1, 4, 44)
    x = torch.randn(number, 1, 4, 44, device=device)
    x_flat = x.flatten().cpu()
    x = np.array(x_flat)

    es = cma.CMAEvolutionStrategy(x, sigma, {'popsize': popsize})

    best_score = -float('inf')
    best_z = None
    best_seq = None

    all_sequences = []
    all_ons = []
    all_offs = []
    all_radios = []

    for gen in range(max_iter):

        solutions = es.ask()
        scores = []

        for s in solutions:

            score, sequences, ons, offs, radios, toehold_switch_sequence, Trigger_rnas, DeltaDeltaG_opens, MFE_selfs  = blackbox_objective_z_tensor(s, diffusion_model, predictor_on, predictor_off, shape=shape)

            df = pd.DataFrame({
                'on': ons,
                'off': offs,
                'radio': radios,
                'sequence': sequences,
                'toehold_switch_sequence': toehold_switch_sequence,
                'Trigger_rnas': Trigger_rnas,
                'DeltaDeltaG_opens': DeltaDeltaG_opens,
                'MFE_selfs':MFE_selfs
                })
            
            # get today's month and day, e.g. "0921"
            today = datetime.now().strftime("%m%d")

            # construct result directory path
            result_dir = f"../result-{today}/"
            os.makedirs(result_dir, exist_ok=True)  # create directory if it does not exist

            # save CSV, with iteration and score in filename
            df.to_csv(os.path.join(result_dir, f"output_opt_iter={gen}_score={score}.csv"), index=False)

            all_sequences += sequences
            all_offs += offs.tolist()
            all_ons += ons.tolist()
            all_radios += radios.tolist()

            scores.append(score)

            # track best score
            if -score > best_score:

                best_score = -score
                # best_z = torch.tensor(s, dtype=torch.float32).reshape(shape).to('cuda')

                # with torch.no_grad():
                #     best_seq = diffusion_model(best_z)

        es.tell(solutions, scores)
        print(f"[Gen {gen}] best average score = {best_score:.4f}")

    # return best_z, best_seq, best_score
    return all_sequences, all_ons, all_offs, all_radios


def divide_lists(ons: list, offs: list) -> list:
    if len(ons) != len(offs):
        raise ValueError("两个列表长度不一致")
    
    result = []
    for on, off in zip(ons, offs):
        if off == 0:
            result.append(float('inf'))  # or use 0, None, np.nan, etc.
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

    on, off, sequence = main_function(promoters_number=promoter_number,opt=opt)

    ons += on.tolist()
    offs += off.tolist()
    sequences += sequence

    df = pd.DataFrame({
    'on': on,
    'off':off,
    'radio':on-off,
    'sequence': sequences
    })

    if opt:
        df_sorted = df.sort_values(by='radio', ascending=False).head(promoter_number)
        df_sorted.to_csv('../result/opt_output.csv', index=False)
         
    else:
        
        # sort by strength in descending order and keep the top 3×promoter_number rows
        df_sorted = df.sort_values(by='radio', ascending=False).head(promoter_number*3)
        df_sorted.to_csv('../result/output.csv', index=False)
