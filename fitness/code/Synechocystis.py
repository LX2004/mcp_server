import csv
import numpy as np
import torch

def text_build_vocab():
    
    dic = [a for a in 'ATCG']
    dic += [a + b for a in 'ATCG' for b in 'ATCG']
    dic += [a + '0' for a in 'ATCG']
    return dic

def Dimer_split_seqs(seq):
    t = text_build_vocab()
    # print('t = ', t)
    # pdb.set_trace()
    ori_result = []
    dim_result = []
    pos_result = []
    
    result = ''

    lens = len(seq)

    for i in range(lens):
        result += ' ' + seq[i].upper()
        ori_result.append(t.index(seq[i].upper()))

    # dimer_encode
    # result += ' '
    # result += 'SEP1'

    seq += '0'
    wt = 2
    for i in range(lens):
        result += ' ' + seq[i:i + wt].upper()
        dim_result.append(t.index(seq[i:i + wt].upper()))
    
    # print('result = ',result)
    
    # pdb.set_trace()

    pos_result += [i for i in range(1, lens + 1)]
    # print('ori_result = ', ori_result)
    # print('dim_result = ', dim_result)
    # print('pos_result = ', pos_result)
    if ori_result[0] < 0:
        # pdb.set_trace()
        print('seq = ', seq)
    
    seq_r = []
    seq_r.append(ori_result)
    seq_r.append(dim_result)
    seq_r.append(pos_result)
    # print('ori lenth = ',len(ori_result))
    # print('dim lenth = ',len(dim_result))
    # print('pos lenth = ',len(pos_result))
    # pdb.set_trace()
    # seq = pd.concat([nuc_seq, pos_seq], axis=0, ignore_index=True)

    return seq_r


def read_data_Synechocystis(df):

    guides = []
    lights = []
    oris = []

    
    for guide, light, ori in zip(df['guide_rna'], df['light'], df['ori']):

        guides.append(guide)
        lights.append(light)
        oris.append(ori)

    return guides, lights, oris


def make_dataset(guides, oris, lights):

    features_array = []
    bios_array = []

    base_choice_ori = ['+', '-']
    base_choice_light = ['100', '300', '0/300']

    for sequence, ori, light in zip(guides, oris, lights):

        # print('!!!!!')

        if len(sequence) != 20:

            print('length = ', len(sequence))
            print('sequence = ',sequence)
            # continue
        
        # one-hot encoding for ori
        if ori in base_choice_ori:

            if ori == '+':
                ori = np.array([1,0])
            
            else:
                ori = np.array([0,1])

        else:
            raise ValueError(f"The input ori string '{ori}' is not in the allowed list and cannot be one-hot encoded.")
            # continue

        # one-hot encoding for light
        # print(type(light))
        if light in base_choice_light:

            if light == '100':
                light = np.array([1,0,0])
            
            elif light == '300':
                light = np.array([0,1,0])

            else :
                light = np.array([0,0,1])

        else:
            raise ValueError(f"The input light string '{light}' is not in the allowed list and cannot be one-hot encoded.")
            # continue
        # print('!!!!!')

        feature = Dimer_split_seqs(sequence)  # all sequences as input
        feature = np.array(feature)
        feature = feature.astype(int)

        # print('!!!!!')

        features_array.append(feature)
        bios_array.append(np.concatenate((light, ori)))
    
    return np.array(features_array), np.array(bios_array)

def encode_list(seqList):

    X_seq=np.array([Dimer_split_seqs(''.join(np.char.join('', sequence))) for sequence in seqList])

    return X_seq

def plot_target_label(seq, bio, model, device): # plot distribution of ground truth and predictions

    # device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    # torch.cuda.set_device(2)
    
    # randomset_=encode_list(seq)
    randomset_ = torch.tensor(seq).to(device)

    bio_  = torch.tensor(bio).to(device)
    df_pred = compute_scaler(model(randomset_, bio_))

    df_pred = np.array(df_pred).squeeze()
    return df_pred

def compute_scaler(model_output):

    model_output = model_output.detach().cpu().numpy()

    # max_reads =  2.5
    # min_reads =  -3.5

    max_reads =  4.7463
    min_reads =  -7.3476

    model_output = min_reads + (max_reads - min_reads) * model_output

    return model_output
