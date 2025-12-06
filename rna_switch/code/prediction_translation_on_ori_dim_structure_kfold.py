import hyperopt
from hyperopt import fmin, tpe, hp, Trials
import torch
from utils import *
from net import predict_transformerv2
from initialize import initialize_weights
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pdb
import os
from sklearn.model_selection import KFold


def build_structure_array(seq: str) -> np.ndarray:
    """
    Build an N×N structural array for an RNA sequence, where each entry
    encodes the base-pairing potential (in terms of hydrogen bond count).

    Args:
        seq (str): input RNA sequence (e.g. 'AUGCGAU...')

    Returns:
        np.ndarray: N×N structural array, encoded as 0, 2, 3
    """
    seq = seq.upper().replace('T', 'U')  # support DNA input by converting T to U
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


def make_dataset_sequences_bio(mRNAs, ons, offs, on_offs, out_label='on', structure=False):

    features_array = []
    labels_array = []
    structures = []

    max_on = max(ons)
    max_off = max(offs)
    max_on_off = max(on_offs)

    min_off = min(offs)
    min_on = min(ons)
    min_on_off = min(on_offs)

    number = 0

    for mRNA, on, off, on_off in zip(mRNAs, ons, offs, on_offs):

        if len(mRNA) != 115:

            print('length = ', len(mRNA))
            print('sequence = ', mRNA)
            pdb.set_trace()

            continue

        feature = Dimer_split_seqs(mRNA)  # use all sequences as input
        feature = np.array(feature)
        feature = feature.astype(int)
        matrix = build_structure_array(seq=mRNA)

        # print(matrix.shape)
        structures.append(matrix)
        # pdb.set_trace()

        features_array.append(feature)

        label_on = (on - min_on) / (max_on - min_on)
        label_off = (off - min_off) / (max_off - min_off)
        label_on_off = (on_off - min_on_off) / (max_on_off - min_on_off)

        if out_label == 'on':

            labels_array.append(label_on)

        elif out_label == 'off':
            labels_array.append(label_off)

        elif out_label == 'on_off':
            labels_array.append(label_on_off)

        else:
            print('flag is error')

        number += 1

    print('number = ', number)

    if structure:
        return np.array(features_array), np.array(structures), np.array(labels_array),

    else:
        return np.array(features_array), np.array(labels_array)


def read_data(filename):

    # Loop1 - Switch  - Loop2 - Stem1 -  AUG  -  Stem2  -  Linker - Post-linker

    import math

    mRNAs = []
    ons = []

    offs = []
    on_offs = []

    df = pd.read_csv(filename)

    number = 0

    for loop1, switch, loop2, stem1, atg, stem2, linker, post_linker, on, off, on_off in zip(
        df['loop1'], df['switch'], df['loop2'], df['stem1'], df['atg'],
        df['stem2'], df['linker'], df['post_linker'], df['ON'], df['OFF'], df['ON_OFF']
    ):

        # convert to float
        on = float(on)
        off = float(off)
        on_off = float(on_off)

        if math.isnan(on) or math.isnan(off) or math.isnan(on_off):

            print(f'on is {on}!!!\noff is {off}!!!\non_off is {on_off}!!!')

            continue

        mRNAs.append(loop1 + switch + loop2 + stem1 + atg + stem2 + linker + post_linker)
        ons.append(on)
        offs.append(off)
        on_offs.append(on_off)

        number += 1

    print('number is ', number)
    return mRNAs, ons, offs, on_offs


# custom dataset class
class CustomDataset(Dataset):
    def __init__(self, features, structures, labels):

        self.features = features
        self.structures = structures
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):

        feature = self.features[idx]
        structure = self.structures[idx]
        label = self.labels[idx]

        return feature, structure, label


def file_detection(file_path):
    # if file does not exist, create it
    if not os.path.exists(file_path):
        # create parent directory if needed
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # create empty file
        with open(file_path, 'w') as f:
            pass

    print(f"文件已确保存在：{file_path}")


def train(params, features_array, structure_array, labels_array):

    patience = 50

    print('params = ', params)

    # store pearson correlation coefficients for cross-validation
    test_pearson_kfold = []

    file_detection(file_path='../result/on_good_record_metric_pearson_ori_dim_structure.txt')
    file_detection(file_path='../result/on_good_record_metric_pearson_mse_ori_dim_structure.txt')
    file_detection(file_path='../result/on_good_record_metric_mse_ori_dim_structure.txt')

    # iterate over each fold
    for fold, (train_indices, val_indices) in enumerate(kf.split(features_array)):

        best_val_loss = float('inf')  # best validation loss
        no_improve_epochs = 0  # epochs without validation improvement

        print(f"Fold {fold + 1}/{k_folds}")

        print('size of train datset is: ', len(train_indices))
        print('size of test datset is: ', len(val_indices))

        # create dataset objects
        train_dataset = CustomDataset(features_array[train_indices], structure_array[train_indices], labels_array[train_indices])
        test_dataset = CustomDataset(features_array[val_indices], structure_array[val_indices], labels_array[val_indices])

        # create data loaders
        train_loader = DataLoader(train_dataset, batch_size=params['train_batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=params['train_batch_size'], shuffle=False)

        # check loader lengths
        print('训练集长度 = ', len(train_loader))
        print('测试集长度 = ', len(test_loader))

        # instantiate model
        print('start compose simple gan model')
        gen = predict_transformerv2.Predict_translation_structure(params=params).to(device)

        initialize_weights(gen)
        print('successful compose simple gan model')

        # define optimizer
        opt_gen = torch.optim.Adam(gen.parameters(), lr=params['train_base_learning_rate'], weight_decay=params['l2_regularization'])
        loss_fc = torch.nn.MSELoss()

        loss_train = []
        loss_test = []

        metric = []
        all_mse = []
        all_r2 = []
        all_spearman_corr = []

        """Start training"""
        for epoch in range(params['train_epochs_num']):

            # adjust learning rate
            if epoch > 0 and epoch % 100 == 0:

                for param_group in opt_gen.param_groups:

                    print('调节学习速率')
                    param_group['lr'] = param_group['lr'] / 2.0

            loss_train_one_epoch = 0
            loss_test_one_epoch = 0

            loss_mse = 0
            loss_pier = 0

            # training phase
            gen.train()

            for data, struc, target in train_loader:

                data = data.to(device)
                target = target.to(device)
                struc = struc.to(device)

                # print('data.shape = ', data.shape)
                # print('target.shape = ', target.shape)
                # print('struc.shape = ', struc.shape)

                output = gen(data, struc)
                output = torch.squeeze(output, dim=1)

                loss_gen = loss_fc(target.float(), output.float())
                loss_pi = loss_pierxun(target=target.float(), output=output.float())

                # print('*****loss_gen = ******',loss_gen)
                loss_gen = loss_gen.float()
                loss_pi = loss_pi.float()

                if loss_kind == 'pearson':
                    loss_all = -loss_pi

                elif loss_kind == 'pearson_mse':
                    loss_all = -loss_pi + loss_gen

                elif loss_kind == 'mse':
                    loss_all = loss_gen

                else:
                    print('输入的损失函数类型有误，请检查！！！')

                opt_gen.zero_grad()
                loss_all.backward()
                opt_gen.step()

                loss_train_one_epoch += loss_all.item()
                loss_mse += loss_gen.item()
                loss_pier += loss_pi.item()

            loss_train.append(loss_train_one_epoch / len(train_loader))

            if epoch % 10 == 0:
                print(
                    f"Epoch[{epoch}/{params['train_epochs_num']}] ****Train loss: {loss_train_one_epoch/len(train_loader):.6f}****MSE loss: {loss_mse/len(train_loader):.6f}****Pierxun loss: {loss_pier/len(train_loader):.6f}"
                )

            # evaluation phase
            gen.eval()

            targets = []
            outputs = []

            for data, struc, target in test_loader:

                data = data.to(device)
                target = target.to(device)
                struc = struc.to(device)

                output = gen(data, struc)
                output = torch.squeeze(output, dim=1)
                loss_gen = loss_fc(target, output)

                targets.append(target.detach().cpu().numpy())
                outputs.append(output.detach().cpu().numpy())

                loss_test_one_epoch += loss_gen.item()

            correlation_coefficient = compute_correlation_coefficient(
                np.concatenate(targets, axis=0), np.concatenate(outputs, axis=0)
            )
            mse, r2, spearman_corr = evaluate_regression_metrics(
                np.concatenate(targets, axis=0), np.concatenate(outputs, axis=0)
            )
            # pdb.set_trace()

            loss_test.append(loss_test_one_epoch / len(test_loader))

            # check if validation loss improved
            if loss_test_one_epoch / len(test_loader) < best_val_loss:
                best_val_loss = loss_test_one_epoch / len(test_loader)

                no_improve_epochs = 0  # reset counter
            else:
                no_improve_epochs += 1

            if epoch % 10 == 0:

                print(
                    f"Epoch[{epoch}/{params['train_epochs_num']}] ****Test loss: {loss_test_one_epoch/len(test_loader):.6f}********test correlation_coefficient:{correlation_coefficient}"
                )

            metric.append(correlation_coefficient)
            all_mse.append(mse)
            all_r2.append(r2)
            all_spearman_corr.append(spearman_corr)

            # save loss and model

            global pcc  # explicitly declare pcc as a global variable
            if correlation_coefficient > pcc:

                pcc = correlation_coefficient

                if loss_kind == 'pearson':
                    torch.save(gen, '../model/on_pearson_structure_{0}_pcc={1:.4f}.pth'.format(epoch, correlation_coefficient))

                elif loss_kind == 'pearson_mse':
                    torch.save(gen, '../model/on_pearson_mse_structure_{0}_pcc={1:.4f}.pth'.format(epoch, correlation_coefficient))

                elif loss_kind == 'mse':
                    torch.save(gen, '../model/on_mse_structure_{0}_pcc={1:.4f}.pth'.format(epoch, correlation_coefficient))

                else:
                    print('损失函数类型出错，请检查！！！！')

            # learning rate decay
            if no_improve_epochs > 0 and no_improve_epochs % 10:

                for param_group in opt_gen.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9

            # early stopping
            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # record metrics
        dict2 = {
            'correlation_coefficient': max(metric),
            'mse': min(all_mse),
            'r2': max(all_r2),
            'spearman_corr': max(all_spearman_corr),
            'min_train_loss': min(loss_train),
            'min_test_loss': min(loss_test),
            'k_fold': fold + 1
        }

        if loss_kind == 'pearson':
            write_good_record(
                dict1=params,
                dict2=dict2,
                file_path='../result/on_good_record_metric_pearson_ori_dim_structure.txt'
            )

        elif loss_kind == 'pearson_mse':
            write_good_record(
                dict1=params,
                dict2=dict2,
                file_path='../result/on_good_record_metric_pearson_mse_ori_dim_structure.txt'
            )

        elif loss_kind == 'mse':
            write_good_record(
                dict1=params,
                dict2=dict2,
                file_path='../result/on_good_record_metric_mse_ori_dim_structure.txt'
            )

        else:
            print('损失函数类型出错，请检查！！！！')

        test_pearson_kfold.append(max(metric))

    return -max(test_pearson_kfold)


# pcc metric
pcc = 0.81

# train(params,train_dataset,test_dataset)
if __name__ == '__main__':

    # data processing
    filename = '/home/liangce/lx/Promoter_mRNA_synthetic/data/Toehold_mRNA_Dataset_clean.csv'
    mRNAs, ons, offs, on_offs = read_data(filename=filename)
    features_array, structure_array, labels_array = make_dataset_sequences_bio(
        mRNAs, ons, offs, on_offs, structure=True
    )

    # K-fold cross validation
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True)

    # GPU device
    params = {
        'device_num': 4,
        'dropout_rate1': 0.5,
        'dropout_rate2': 0.2,
        'dropout_rate_fc': 0.48,
        'embedding_dim1': 128,
        'embedding_dim2': 128,
        'fc_hidden1': 144,
        'fc_hidden2': 8,
        'hidden_dim1': 64,
        'hidden_dim2': 512,
        'l2_regularization': 5e-05,
        'latent_dim': 512,
        'num_head1': 8,
        'num_head2': 16,
        'seq_len': 115,
        'train_base_learning_rate': 0.0014,
        'train_batch_size': 512,
        'train_epochs_num': 500,
        'transformer_num_layers1': 3,
        'transformer_num_layers2': 10
    }
    # params = {'device_num': 2, 'dropout_rate1': 0.3258494549467406, 'dropout_rate2': 0.2974783660130027, 'dropout_rate_fc': 0.3134874750986153, 'embedding_dim1': 64, 'embedding_dim2': 256, 'fc_hidden1': 109, 'fc_hidden2': 56, 'hidden_dim1': 1024, 'hidden_dim2': 256, 'l2_regularization': 5e-05, 'latent_dim1': 64, 'latent_dim2': 256, 'num_head1': 8, 'num_head2': 8, 'seq_len': 20, 'train_base_learning_rate': 0.0010350836441350173, 'train_batch_size': 512, 'train_epochs_num': 500, 'transformer_num_layers1': 4, 'transformer_num_layers2': 8}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(params['device_num'])
    print('device =', device)

    # choose loss type
    # loss_kind = ['pearson', 'pearson_mse', 'mse']
    loss_kind = 'mse'

    # single trial run
    train(params, features_array=features_array, structure_array=structure_array, labels_array=labels_array)

    """Start hyperparameter search"""
    # hyperparameter search space
    space = {

        'train_batch_size': hp.choice('train_batch_size', [512]),
        'seq_len': hp.choice('seq_len', [115]),
        'device_num': hp.choice('device_num', [4]),
        'train_epochs_num': hp.choice('train_epochs_num', [500]),

        'train_base_learning_rate': hp.loguniform('train_base_learning_rate', -7, -4),

        'dropout_rate1': hp.uniform('dropout_rate1', 0.1, 0.5),
        'dropout_rate2': hp.uniform('dropout_rate2', 0.1, 0.5),
        'dropout_rate_fc': hp.uniform('dropout_rate_fc', 0.1, 0.5),

        'transformer_num_layers1': hp.randint('transformer_num_layers1', 1, 12),
        'transformer_num_layers2': hp.randint('transformer_num_layers2', 1, 12),

        # 'l2_regularization': hp.loguniform('l2_regularization', -8, -2),
        'l2_regularization': hp.choice('l2_regularization', [5e-5, 2e-5, 5e-6]),

        'num_head1': hp.choice('num_head1', [2, 4, 8, 16]),
        'num_head2': hp.choice('num_head2', [2, 4, 8, 16]),

        'hidden_dim1': hp.choice('hidden_dim1', [64, 128, 256, 512, 1024]),
        'latent_dim': hp.choice('latent_dim', [64, 128, 256, 512]),
        'embedding_dim1': hp.choice('embedding_dim1', [64, 128, 256, 512]),

        'hidden_dim2': hp.choice('hidden_dim2', [128, 256, 512, 1024]),
        'embedding_dim2': hp.choice('embedding_dim2', [64, 128, 256, 512]),

        'fc_hidden1': hp.randint('fc_hidden1', 64, 256),
        'fc_hidden2': hp.randint('fc_hidden2', 8, 64)
    }

    # create Trials to track optimization
    trials = Trials()

    # wrap train function as hyperopt objective
    objective = lambda params: train(
        params,
        features_array=features_array,
        structure_array=structure_array,
        labels_array=labels_array
    )
    # run optimization
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=1000, trials=trials)

    # print best hyperparameters
    print('最佳参数:', best)
