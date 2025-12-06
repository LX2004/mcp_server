
import os
import pandas as pd
from io import StringIO
import numpy as np
from utils import *
from net import predict_transformerv2

def compute_scaler(model_output, max_reads, min_reads):

    model_output = model_output.detach().cpu().numpy()

    max_reads_mddel =  np.max(model_output)
    min_reads_mddel =  np.min(model_output)
    
    if max_reads_mddel != min_reads_mddel:

        model_output = (model_output - min_reads_mddel) / (max_reads_mddel - min_reads_mddel)

    model_output = min_reads + (max_reads - min_reads) * model_output

    return model_output

def encode_list(seqList):

    X_seq=np.array([Dimer_split_seqs(''.join(np.char.join('', sequence))) for sequence in seqList])

    return X_seq


def compute_efficiency(seq, model, max_reads = 11.18553192156916, min_reads = -7.686416791646049, device=torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')): 

    randomset_=encode_list(seq)
    randomset_ = torch.tensor(randomset_).to(device)

    df_pred = compute_scaler(model(randomset_),max_reads=max_reads,min_reads=min_reads)
    df_pred = np.array(df_pred).squeeze()

    return df_pred

def make_dataset_for_efficiency(guides):

    seq = []

    split_sequence = [char for char in guides]
    seq.append(split_sequence)

    return np.array(seq)


def Quickly_predict_efficiency(sgRNA):
    

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(2)
    model = torch.load('../model/effiency_prediction.pth', weights_only=False).to(device)
    
    seq = make_dataset_for_efficiency(sgRNA)

    print('seq.shape = ', seq.shape)

    predict_result = compute_efficiency(seq, model)
    
    return predict_result



if __name__ == "__main__":
 

    efficiency = Quickly_predict_efficiency(sgRNA)

