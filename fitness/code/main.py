import os
import pandas as pd
from io import StringIO
import numpy as np
from utils import *
from net import predict_transformerv2
import Synechocystis
from scipy.stats import pearsonr
# from bad_seed import *
# from calculate_fitness import compute_main

def read_data_Ecoli(df):

    guides = []
    essentials = []

    oris = []
    codings = []
    
    for guide, essential, ori, coding in zip(df['guide_rna'], df['essential'], df['ori'], df['coding']):

        guides.append(guide)
        essentials.append(essential)

        oris.append(ori)
        codings.append(coding)

    return guides, essentials, oris, codings


def plot_target_label_feature(seq, bio, model, max_reads = 1.71, min_reads = -3.5, device=torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')):

    randomset_ = torch.tensor(seq).to(device)

    bio_  = torch.tensor(bio).to(device)
    df_pred = compute_scaler(model(randomset_, bio_), max_reads=max_reads, min_reads=min_reads)

    df_pred = np.array(df_pred).squeeze()
    return df_pred


def read_data_Bacillus(df):

    guides = []
    essentials = []
    
    for guide, essential in zip(df['guide_rna'], df['essential']):

        guides.append(guide)
        essentials.append(essential)

    return guides, essentials

def read_data_staphy(df):

    guides = []
    essentials = []
    
    for guide, essential in zip(df['guide_rna'], df['essential']):

        guides.append(guide)
        essentials.append(essential)

    return guides, essentials


def read_data_E_limosum(df):
    return df['guide_rna'].tolist(), df['condition'].tolist()

def make_dataset_bacillus_bio(guides, essentials):

    features_array = []
    bios_array = []
 
    for sequence, essential in zip(guides, essentials):

        if len(sequence) < 20:

            print('length = ', len(sequence))
            print('sequence = ', sequence)
            continue
        
        essential = str(essential)

        if essential == 'True' or 'TRUE':
        
            ori = np.array([1,0])
        
        elif essential == 'False' or 'FALSE':
            ori = np.array([0,1])
            
        else:
            print(f"The input Essential = {essential} is not in the allowed list for one-hot encoding.")
            continue

        feature = Dimer_split_seqs(sequence)
        feature = np.array(feature)
        feature = feature.astype(int)

        features_array.append(feature)
        bios_array.append(ori)
    
    return np.array(features_array), np.array(bios_array)


def make_dataset_staphy_bio(guides, essentials):

    features_array = []
    bios_array = []
 
    for sequence, essential in zip(guides, essentials):

        if len(sequence) < 20:

            print('length = ', len(sequence))
            print('sequence = ', sequence)
            continue
        
        essential = str(essential)

        if essential == 'True' or 'TRUE':
        
            ori = np.array([1,0])
        
        elif essential == 'False' or 'FALSE':
            ori = np.array([0,1])
            
        else:
            print(f"The input Essential = {essential} is not in the allowed list for one-hot encoding.")
            continue

        feature = Dimer_split_seqs(sequence)
        feature = np.array(feature)
        feature = feature.astype(int)

        features_array.append(feature)
        bios_array.append(ori)
    
    return np.array(features_array), np.array(bios_array)

def make_dataset_E_limosum(guides, conditions):

    features_array = []
    bios_array = []

    print(guides)
    print(conditions)

    base_conditions = ['GP', 'CP', 'SynP']

    for sequence, condition in zip(guides, conditions):

        if len(sequence) < 20:

            print('length = ', len(sequence))
            print('sequence = ', sequence)
            continue
        
        if condition in base_conditions:

            if condition == 'GP':
                ori = np.array([1,0,0])
            
            elif condition == 'CP':
                ori = np.array([0,1,0])
            
            else: 
                ori = np.array([0,0,1])

        else:
            print("The input condition string is not in the given list, and one-hot encoding cannot be performed.")
            continue

        feature = Dimer_split_seqs(sequence[-20:])
        feature = np.array(feature)
        feature = feature.astype(int)

        features_array.append(feature)
        bios_array.append(ori)
    
    return np.array(features_array), np.array(bios_array)



def make_dataset_for_find_prometer(guides, essentials, oris, codings):

    bio = []
    seq = []

    for sequence,  essential, ori, coding in zip(guides, essentials, oris, codings):

        split_sequence = [char for char in sequence]
        seq.append(split_sequence)

        essential_feature = encode_essential(essential)
        ori_feature = encode_ori(ori)
        coding_feature = encode_coding(coding)

        biofeature = np.concatenate((essential_feature, ori_feature, coding_feature))
        bio.append(biofeature)

    return np.array(seq), np.array(bio)

def compute_scaler(model_output, max_reads = 1.71, min_reads = -3.5):

    model_output = model_output.detach().cpu().numpy()

    max_reads_mddel = np.max(model_output)
    min_reads_mddel = np.min(model_output)
    
    if max_reads_mddel != min_reads_mddel:

        model_output = (model_output - min_reads_mddel) / (max_reads_mddel - min_reads_mddel)

    model_output = min_reads + (max_reads - min_reads) * model_output

    return model_output

def encode_list(seqList):

    X_seq = np.array([Dimer_split_seqs(''.join(np.char.join('', sequence))) for sequence in seqList])

    return X_seq

def plot_target_label(seq, bio, model, max_reads = 1.71, min_reads = -3.5):

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(2)
    
    randomset_ = encode_list(seq)
    randomset_ = torch.tensor(randomset_).to(device)

    bio_  = torch.tensor(bio).to(device)
    df_pred = compute_scaler(model(randomset_, bio_), max_reads=max_reads, min_reads=min_reads)

    df_pred = np.array(df_pred).squeeze()
    return df_pred


def prediction_Ecoli(inputCsv):

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(2)
    model = torch.load('../model/prediction_model_Ecoli.pth', weights_only=False).to(device)

    df = pd.read_csv(StringIO(inputCsv))
    print('df', df)

    guides, essentials, oris, codings = read_data_Ecoli(df)
    seq, bio = make_dataset_for_find_prometer(guides, essentials, oris, codings)
    print('seq.shape = ', seq.shape)

    predictions = plot_target_label(seq, bio, model)
    predictions = predictions.tolist()

    df["prediction_fitness"] = predictions

    df.to_csv("../result/Ecoli_predictions.csv", index=False)

    pccGraph = [{"label": label, "x": round(y, 4), "y": round(x, 4)} for x, y, label in zip(df["prediction_fitness"], df["fitness"], df["guide_rna"])]

    print(df["guide_rna"])

    if 'fitness' in df.columns:
        pcc, p_value = pearsonr(df["fitness"], df["prediction_fitness"])

    csv = df.to_csv(index=False)

    csv = csv.replace('True', 'TRUE').replace('False', 'FALSE')
    print('csv = \n', csv)

    response = jsonify({
        'csv': csv,
        'pcc': pcc,
        'pccGraph': pccGraph
    })

    return response

def prediction_Synechocystis(inputCsv):
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(2)
    model = torch.load('../model/prediction_model_Synechocystis.pth', weights_only=False).to(device)

    df = pd.read_csv(StringIO(inputCsv))
    print('load data: \n', df)

    guides, lights, oris = Synechocystis.read_data_Synechocystis(df)
    lights = [str(x) for x in lights]
    seq, bio = Synechocystis.make_dataset(guides, oris, lights)

    print('seq.shape = ', seq.shape)

    predictions = Synechocystis.plot_target_label(seq, bio, model, device=device)
    predictions = predictions.tolist()

    df["prediction_fitness"] = predictions

    pccGraph = [{"label": label, "x": round(y, 4), "y": round(x, 4)} for x, y, label in zip(df["prediction_fitness"], df["fitness"], df["guide_rna"])]

    if 'fitness' in df.columns:
        pcc, p_value = pearsonr(df["fitness"], df["prediction_fitness"])

    csv = df.to_csv(index=False)

    csv = csv.replace('True', 'TRUE').replace('False', 'FALSE')
    print('csv = \n', csv)

    response = jsonify({
        'csv': csv,
        'pcc': pcc,
        'pccGraph': pccGraph
    })

    return response

def load_csv_as_string(filename, nrows=128):
    df = pd.read_csv(filename, nrows=nrows)
    csv_string = df.to_csv(index=False)
    return csv_string


def prediction_Bacillus(inputCsv):

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(2)
    model = torch.load('../model/prediction_model_Bacillus.pth', weights_only=False).to(device)

    df = pd.read_csv(StringIO(inputCsv))
    print('df', df)

    guides, essentials = read_data_Bacillus(df)

    print(' 1111 ')
    seq, bio = make_dataset_bacillus_bio(guides, essentials)

    print('seq.shape = ', seq.shape)

    predictions = plot_target_label_feature(seq, bio, model, max_reads=1.157, min_reads=-0.2341, device=device)
    predictions = predictions.tolist()
    df["prediction_fitness"] = predictions

    pccGraph = [{"label": label, "x": round(y, 4), "y": round(x, 4)} for x, y, label in zip(df["prediction_fitness"], df["fitness"], df["guide_rna"])]

    if 'fitness' in df.columns:
        pcc, p_value = pearsonr(df["fitness"], df["prediction_fitness"])

    csv = df.to_csv(index=False)

    csv = csv.replace('True', 'TRUE').replace('False', 'FALSE')
    print('csv = \n', csv)

    response = jsonify({
        'csv': csv,
        'pcc': pcc,
        'pccGraph': pccGraph
    })

    return response


def read_data_clean_staphy(df):
    import math
    guides = []
    fit18s = []

    essentials = []
    number = 0
    for variant_guide, essential, fitness in zip(df['guide_rna'], df['essential'], df['fitness']):
        
        fitness = float(fitness)
        essential = str(essential)

        if math.isnan(fitness) or len(variant_guide) < 20:
            print(f'fitness is {fitness}!!!')
            continue
        if not essential in ['True', 'False', False, True, 'FALSE', 'TRUE']:
            continue

        guides.append(variant_guide.upper())
        fit18s.append(fitness)
        essentials.append(essential)

        number += 1
    

    return guides, fit18s, essentials 


def make_dataset_sequences_bio_clean_staphy(guides, fit18s, essentials):

    features_array = []
    bios_array = []
    labels_array = []

    fit18s = np.array(fit18s)
    max_reads = np.max(fit18s) 
    min_reads = np.min(fit18s)

    print('max_reads = ', max_reads)
    print('min_reads = ', min_reads)

    number = 0
 

    for sequence, score, essential in zip(guides, fit18s, essentials):

        if len(sequence) < 20:

            print('length = ', len(sequence))
            print('sequence = ', sequence)
            continue
        
        essential = str(essential)

        if essential == 'True' or essential == True or essential == 'TRUE':
        
            ori = np.array([1,0])
        
        elif essential == 'False' or essential == False or essential == 'FALSE':
            ori = np.array([0,1])
            
        else:
            print(f"The input Essential = {essential} is not in the allowed list for one-hot encoding.")
            continue

        feature = Dimer_split_seqs(sequence)
        feature = np.array(feature)
        feature = feature.astype(int)

        label = (score - min_reads)/(max_reads -  min_reads)

        features_array.append(feature)
        bios_array.append(ori)
        labels_array.append(label)

        number += 1
    print('number = ', number)
    
    return np.array(features_array), np.array(labels_array), np.array(bios_array)

def prediction_Staphyloccus(inputCsv):

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(2)
    model = torch.load('../model/prediction_model_Staphyloccus.pth', weights_only=False).to(device)

    df = pd.read_csv(StringIO(inputCsv))
    print('df', df)

    guides, fit18s, essentials = read_data_clean_staphy(df)
    df['fitness'] = fit18s
    features_array, labels_array, biofeatures_array = make_dataset_sequences_bio_clean_staphy(guides, fit18s, essentials)
    output = model(torch.tensor(features_array).to(device), torch.tensor(biofeatures_array).to(device))
    print('output = ', output)
    prediction = compute_scaler(output, max_reads=2.43, min_reads=-9.74)
    print('000')
    predictions = np.squeeze(prediction)

    print('1111')
    predictions = predictions.tolist()
    print('222')
    df["prediction_fitness"] = predictions

    pccGraph = [{"label": label, "x": round(y, 4), "y": round(x, 4)} for x, y, label in zip(df["prediction_fitness"], df["fitness"], df["guide_rna"])]
    if 'fitness' in df.columns:
        pcc, p_value = pearsonr(df["fitness"], df["prediction_fitness"])

    csv = df.to_csv(index=False)

    csv = csv.replace('True', 'TRUE').replace('False', 'FALSE')
    print('csv = \n', csv)

    response = jsonify({
        'csv': csv,
        'pcc': pcc,
        'pccGraph': pccGraph
    })

    return response

def prediction_E_limosum(inputCsv):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(1)
    model = torch.load('../model/prediction_model_E_limosum.pth', weights_only=False).to(device)

    df = pd.read_csv(StringIO(inputCsv))

    guides, conditions = read_data_E_limosum(df)

    seq, bio = make_dataset_E_limosum(guides, conditions)
    print(df.columns)
    
    predictions = plot_target_label_feature(seq, bio, model, max_reads=4.5, min_reads=-2.3225, device=device)
    print('prediction = ', predictions)
    predictions = predictions.tolist()

    df["prediction_fitness"] = predictions

    pccGraph = [{"label": label, "x": round(y, 4), "y": round(x, 4)} for x, y, label in zip(df["prediction_fitness"], df["fitness"], df["guide_rna"])]

    if 'fitness' in df.columns:
        pcc, p_value = pearsonr(df["fitness"], df["prediction_fitness"])

    csv = df.to_csv(index=False)

    csv = csv.replace('True', 'TRUE').replace('False', 'FALSE')
    print('csv = \n', csv)

    response = jsonify({
        'csv': csv,
        'pcc': pcc,
        'pccGraph': pccGraph
    })

    return response


def Quickly_predict_fitness_E_coli(sgRNA):
    
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(2)
    model = torch.load('../model/prediction_model_Ecoli.pth', weights_only=False).to(device)
    
    guides = []
    essentials = []
    oris = []
    codings = [] 
    
    
    for ori in ['+','-']:
        for coding in ['FALSE','TRUE']:
            for essential in ['FALSE','TRUE']:
                
                guides.append(sgRNA)
                essentials.append(essential)
                codings.append(coding)
                oris.append(ori)
                
    seq, bio = make_dataset_for_find_prometer(guides, essentials, oris, codings)
    print('seq.shape = ', seq.shape)

    predictions = plot_target_label(seq, bio, model)
    predict_result = np.mean(predictions)
    
    return predict_result


def Quickly_predict_fitness_Synechocystis(sgRNA):
    
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(2)
    model = torch.load('../model/prediction_model_Synechocystis.pth', weights_only=False).to(device)

    guides = []
    lights = []
    oris = []
 
    for ori in ['+','-']:
        for light in ['100', '300', '0/300']:
                            
            guides.append(sgRNA)
            lights.append(light)
            oris.append(ori)
    
    seq, bio = Synechocystis.make_dataset(guides, oris, lights)

    predictions = Synechocystis.plot_target_label(seq, bio, model, device=device)
    predict_result = np.mean(predictions)
    
    return predict_result


def Quickly_predict_fitness_Bacillus_subtillis(sgRNA):
    
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(2)
    model = torch.load('../model/prediction_model_Bacillus.pth', weights_only=False).to(device)
    
    guides = []
    essentials = []
    
    for essential in ['FALSE','TRUE']:
        
        guides.append(sgRNA)
        essentials.append(essential)
        
    print('0000')

    seq, bio = make_dataset_bacillus_bio(guides, essentials)
    print('1111')

    randomset_ = torch.tensor(seq).to(device)
    bio_  = torch.tensor(bio).to(device)
    print('22222')
    
    max_reads = 1.157
    min_reads = -0.2341
    
    model_output = model(randomset_, bio_)
    print('4444')
    
    df_pred = min_reads + (max_reads - min_reads) * model_output
    print('5555')
    print('df_pred = ', df_pred)

    predictions = df_pred.cpu().detach().numpy().squeeze()
    print('6666')
    
    print('predictions = ', predictions)

    predict_result = np.mean(predictions)
    
    return predict_result


def Quickly_predict_fitness_Staphyloccus(sgRNA):
    
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(2)
    model = torch.load('../model/prediction_model_Staphyloccus.pth', weights_only=False).to(device)

    guides = []
    essentials = []
    
    for essential in ['FALSE','TRUE']:
        
        guides.append(sgRNA)
        essentials.append(essential)
        
    fit18s = [i for i in range(0, len(guides))]
        
    features_array, labels_array, biofeatures_array = make_dataset_sequences_bio_clean_staphy(guides, fit18s, essentials)
    output = model(torch.tensor(features_array).to(device), torch.tensor(biofeatures_array).to(device))
    prediction = compute_scaler(output, max_reads=2.43, min_reads=-9.74)
    predictions = np.squeeze(prediction)
    
    predict_result = np.mean(predictions)
    
    return predict_result
    

def Quickly_predict_fitness_E_limosum(sgRNA):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(1)
    model = torch.load('../model/prediction_model_E_limosum.pth', weights_only=False).to(device)

    guides = []
    conditions = []
    
    for condition in ['GP','CP', 'SynP']:
        
        guides.append(sgRNA)
        conditions.append(condition)
        
    seq, bio = make_dataset_E_limosum(guides, conditions)

    predictions = plot_target_label_feature(seq, bio, model, max_reads=4.5, min_reads=-2.3225, device=device)
    predict_result = np.mean(predictions)
    
    return predict_result
    
def Quickly_prediction_fitness(name, gRNA):

    if name == 'E_coli':

        prediction_result = Quickly_predict_fitness_E_coli(gRNA[-20:])        
        return f'E_coli  gRNA:{gRNA}, fitness:{prediction_result}'

    elif name == 'Synechocystis':

        prediction_result = Quickly_predict_fitness_Synechocystis(gRNA[-20:])
        return f'Synechocystis  gRNA:{gRNA}, fitness:{prediction_result}'

    elif name == 'Bacillus_subtillis':

        prediction_result = Quickly_predict_fitness_Bacillus_subtillis(gRNA[-20:])
        return f'Bacillus_subtillis  gRNA:{gRNA}, fitness:{prediction_result}'

    elif name == 'Staphyloccus':

        prediction_result = Quickly_predict_fitness_Staphyloccus(gRNA[-20:])
        return f'Staphyloccus  gRNA:{gRNA}, fitness:{prediction_result}'
    
    elif name == 'E_limosum':

        prediction_result = Quickly_predict_fitness_E_limosum(gRNA[-20:])
        return f'E_limosum  gRNA:{gRNA}, fitness:{prediction_result}'

    else: 
        valid_species = ["E_coli", "Cyanobacteria", "Staphylococcus", "E_limosum", "Bacillus_subtillis"]
        raise ValueError(f"Invalid species name: '{name}' is not in the supported list. Available options are: {valid_species}")
    


if __name__ == '__main__':
    
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(2)
    print('device =', device)
    
    name = "E_coli"
    gRNA = "AAAAAACGTATTCGCTTGCA"
    
    a = Quickly_prediction_fitness(name, gRNA)    
    print(a)
