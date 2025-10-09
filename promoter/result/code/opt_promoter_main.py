import os
import argparse
import torch
import numpy as np
import pandas as pd
from utils import *
import script_utils
from utils_prediction import *
import cma

def encode_list(seq_list):
    X_seq = np.array([Dimer_split_seqs(sequence) for sequence in seq_list])
    return X_seq

def compute_scaler(model_output, flag=' '):
    model_output = model_output.detach().cpu().numpy()
    if flag == 'wx':
        max_reads = 12.58829790141715
        min_reads = -8.63820038439808
        model_output = min_reads + (max_reads - min_reads) * model_output
        return np.power(2, model_output)
    else:
        max_reads = 6.440191125726022
        min_reads = 0.9380190974762103
        model_output = min_reads + (max_reads - min_reads) * model_output
        return np.power(10, model_output)

def create_argparser(promoters_number):
    defaults = dict(
        num_images=promoters_number,
        device=device,
        schedule_low=1e-4,
        schedule_high=0.02,
        out_init_conv_padding=1,
    )
    defaults.update(script_utils.diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--save_dir", type=str)
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser

def main_function(promoters_number, opt=False):
    args = create_argparser(promoters_number=promoters_number).parse_args()

    model_path = '../model/generate_model.pth'
    diffusion = script_utils.get_diffusion_from_args(args).to(device)
    diffusion.load_state_dict(torch.load(model_path, weights_only=False))

    prediction_model = torch.load('../model/prediction_model.pth', weights_only=False).to(device)

    sequences = []
    samples = diffusion.sample(args.num_images, device)
    samples = samples.squeeze(dim=1)
    samples = samples.to('cpu').detach().numpy()

    for j in range(samples.shape[0]):
        decoded_sequence = decode_one_hot(samples[j])
        sequences.append(decoded_sequence)

    randomset_ = encode_list(sequences)
    randomset_ = torch.tensor(randomset_).to(device)
    strengths = compute_scaler(prediction_model(randomset_))
    strengths = np.array(strengths).squeeze()
    sequences = np.array(sequences)
    strengths = strengths.astype(int)

    if opt:
        all_sequences, all_strengths = optimize_tensor_input_with_cmaes(
            diffusion_model=diffusion,
            predictor=prediction_model,
            number=promoters_number,
            sigma=0.5,
            max_iter=30,
            popsize=promoters_number,
        )
        return all_sequences, all_strengths
    else:
        return strengths, sequences

def blackbox_objective_z_tensor(z_flat_np, diffusion_model, predictor, shape=(512, 1, 4, 50)):
    sequences = []
    z_tensor = torch.tensor(z_flat_np, dtype=torch.float32).reshape(shape).to(device)
    samples = diffusion_model.sample_opt(x=z_tensor, batch_size=shape[0], device=device)
    samples = samples.squeeze(dim=1)
    samples = samples.to('cpu').detach().numpy()

    for j in range(samples.shape[0]):
        decoded_sequence = decode_one_hot(samples[j])
        sequences.append(decoded_sequence)

    randomset_ = encode_list(sequences)
    randomset_ = torch.tensor(randomset_).to(device)
    strengths = compute_scaler(predictor(randomset_))
    strengths = np.array(strengths).squeeze()
    sequences = np.array(sequences)
    strengths = strengths.astype(int)
    avg_strength = strengths.mean()
    return -avg_strength, sequences, strengths

def tensor_to_promoter(output_tensor):
    samples = output_tensor.squeeze(dim=1)
    samples = samples.to('cpu').detach().numpy()
    sequences = []
    for i in range(samples.shape[0]):
        decoded_sequence = decode_one_hot(samples[i])
        sequences.append(decoded_sequence)
    return sequences

def optimize_tensor_input_with_cmaes(
    diffusion_model,
    predictor,
    number=512,
    sigma=0.5,
    max_iter=5,
    popsize=512,
):
    shape = (number, 1, 4, 50)
    x = torch.randn(number, 1, 4, 50, device=device)
    x_flat = x.flatten().cpu().numpy()
    es = cma.CMAEvolutionStrategy(x_flat, sigma, {'popsize': popsize})

    best_score = -float('inf')
    best_z = None

    all_sequences = []
    all_strengths = []

    for gen in range(max_iter):
        solutions = es.ask()
        scores = []
        for s in solutions:
            score, sequences, strengths = blackbox_objective_z_tensor(s, diffusion_model, predictor, shape=shape)
            all_sequences += sequences.tolist() if isinstance(sequences, np.ndarray) else sequences
            all_strengths += strengths.tolist() if isinstance(strengths, np.ndarray) else strengths
            scores.append(score)
            if -score > best_score:
                best_score = -score
                best_z = torch.tensor(s, dtype=torch.float32).reshape(shape).to(device)
        es.tell(solutions, scores)

    return all_sequences, all_strengths

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == '__main__':
    opt = False
    promoter_number = 16
    strengths = []
    sequences = []

    for _ in range(3):
        strength, sequence = main_function(promoters_number=promoter_number, opt=opt)
        strengths += strength.tolist()
        sequences += sequence.tolist()

    df = pd.DataFrame({'strength': strengths, 'sequence': sequences})
    print(df)

    if opt:
        df_sorted = df.sort_values(by='strength', ascending=False).head(promoter_number)
        df_sorted.to_csv('../result/opt_output.csv', index=False)
    else:
        df_sorted = df.sort_values(by='strength', ascending=False).head(promoter_number * 3)
        df_sorted.to_csv('../result/output_3072.csv', index=False)
