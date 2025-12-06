import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import pdb
import torch
import torch.nn as nn
from net.Transformer_encoder import Predict_encoder


class ResidualBlock(nn.Module):
    def __init__(self, num_channels, kernel_size, padding):
        super(ResidualBlock, self).__init__()
        self.num_channels = num_channels

        # two convolutional layers
        self.conv1 = nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, padding=padding)
        self.batch_norm = nn.BatchNorm1d(num_channels)

        self.ac = nn.LeakyReLU()

    def forward(self, x):
        res = x
        for _ in range(2):
            res = self.conv1(res)
            res = self.batch_norm(res)
            res = self.ac(res)

            res = self.conv2(res)
            res = self.batch_norm(res)

        return x + res


class CrossAttention(nn.Module):
    def __init__(self, in_dim, num_heads, k_dim, v_dim):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        # linear projections for multi-head Q, K, V
        self.proj_q1 = nn.Linear(in_dim, num_heads * k_dim)
        self.proj_k2 = nn.Linear(in_dim, num_heads * k_dim)
        self.proj_v2 = nn.Linear(in_dim, num_heads * v_dim)

        # output layer
        self.strength_output = nn.Linear(num_heads * v_dim, in_dim)

    def forward(self, x1, x2, mask=None):
        # x1 is query, x2 is key and value
        x1 = x1.unsqueeze(1)  # (batch_size, 1, in_dim)
        x2 = x2.unsqueeze(1)  # (batch_size, 1, in_dim)
        batch_size, seq_len1, in_dim1 = x1.size()
        seq_len2 = x2.size(1)

        # project to q1, k2, v2
        q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        # attention scores
        attention = torch.matmul(q1, k2) / self.k_dim ** 0.5

        # optional mask
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)

        # attention weights
        attention = F.softmax(attention, dim=-1)

        # weighted sum
        hin = torch.matmul(attention, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1).squeeze(1)

        # output projection
        strength_output = self.strength_output(hin).squeeze(1)

        return strength_output


class Predict_transformer_cross_attention(torch.nn.Module):
    def __init__(self, params):
        super(Predict_transformer_cross_attention, self).__init__()

        self.dropout_rate_fc = params['dropout_rate_fc']
        self.relu = nn.ReLU()

        self.trans_ori_pos = Predict_encoder(
            nhead=params['num_head1'],
            layers=params['transformer_num_layers1'],
            hidden_dim=params['hidden_dim1'],
            latent_dim=params['latent_dim'],
            embedding_dim=params['embedding_dim1'],
            seq_len=params['seq_len'],
            probs=params['dropout_rate1'],
            device='cuda'
        )
        self.trans_dim_pos = Predict_encoder(
            nhead=params['num_head2'],
            layers=params['transformer_num_layers2'],
            hidden_dim=params['hidden_dim2'],
            latent_dim=params['latent_dim'],
            embedding_dim=params['embedding_dim2'],
            seq_len=params['seq_len'],
            probs=params['dropout_rate2'],
            device='cuda'
        )

        # embeddings
        self.embedding_ori = torch.nn.Embedding(100, params['embedding_dim1'])
        self.embedding_dim = torch.nn.Embedding(100, params['embedding_dim2'])

        # cross attention layers
        self.cross_ori_dim = CrossAttention(
            in_dim=params['latent_dim'],
            num_heads=params['num_head1'],
            k_dim=params['hidden_dim1'],
            v_dim=params['hidden_dim1']
        )
        self.cross_dim_ori = CrossAttention(
            in_dim=params['latent_dim'],
            num_heads=params['num_head1'],
            k_dim=params['hidden_dim2'],
            v_dim=params['hidden_dim2']
        )

        self.ac = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=self.dropout_rate_fc)

        self.final_fc1 = nn.Linear(params['latent_dim'], params['fc_hidden1'])
        self.final_fc2 = nn.Linear(params['fc_hidden1'], params['fc_hidden2'])
        self.final_fc3 = nn.Linear(params['fc_hidden2'], 1)

    def forward(self, X):
        x = X.to(torch.int)

        input_ori = x[:, 0, :]
        input_dim = x[:, 1, :]

        embeded_ori = self.embedding_ori(input_ori)
        embeded_dim = self.embedding_dim(input_dim)

        ori_pos = self.trans_ori_pos(embeded_ori)
        dim_pos = self.trans_dim_pos(embeded_dim)

        # cross-attention-based fusion
        output = self.cross_ori_dim(ori_pos, dim_pos) + self.cross_dim_ori(dim_pos, ori_pos)

        output = self.final_fc1(output)
        output = self.ac(output)
        output = self.dropout(output)

        output = self.final_fc2(output)
        output = self.ac(output)

        output = self.final_fc3(output)

        return self.relu(output)


class Predict_transformer(torch.nn.Module):
    def __init__(self, params):
        super(Predict_transformer, self).__init__()

        self.dropout_rate_fc = params['dropout_rate_fc']
        self.relu = nn.ReLU()

        self.trans_ori_pos = Predict_encoder(
            nhead=params['num_head1'],
            layers=params['transformer_num_layers1'],
            hidden_dim=params['hidden_dim1'],
            latent_dim=params['latent_dim1'],
            embedding_dim=params['embedding_dim1'],
            seq_len=params['seq_len'],
            probs=params['dropout_rate1'],
            device='cuda'
        )
        self.trans_dim_pos = Predict_encoder(
            nhead=params['num_head2'],
            layers=params['transformer_num_layers2'],
            hidden_dim=params['hidden_dim2'],
            latent_dim=params['latent_dim2'],
            embedding_dim=params['embedding_dim2'],
            seq_len=params['seq_len'],
            probs=params['dropout_rate2'],
            device='cuda'
        )

        self.embedding_ori = torch.nn.Embedding(100, params['embedding_dim1'])
        self.embedding_dim = torch.nn.Embedding(100, params['embedding_dim2'])

        self.ac = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=self.dropout_rate_fc)

        self.final_fc1 = nn.Linear(params['latent_dim1'] + params['latent_dim2'], params['fc_hidden1'])
        self.final_fc2 = nn.Linear(params['fc_hidden1'], params['fc_hidden2'])
        self.final_fc3 = nn.Linear(params['fc_hidden2'], 1)

    def forward(self, X):
        x = X.to(torch.int)

        input_ori = x[:, 0, :]
        input_dim = x[:, 1, :]

        embeded_ori = self.embedding_ori(input_ori)
        embeded_dim = self.embedding_dim(input_dim)

        ori_pos = self.trans_ori_pos(embeded_ori)
        dim_pos = self.trans_dim_pos(embeded_dim)

        output = torch.cat((ori_pos, dim_pos), dim=-1)

        output = self.final_fc1(output)
        output = self.ac(output)
        output = self.dropout(output)

        output = self.final_fc2(output)
        output = self.ac(output)

        output = self.final_fc3(output)

        return self.relu(output)


class transformer_ont_biofeat_classification(torch.nn.Module):
    """
    Transformer + CNN + biological features classification model.
    Inspired by:
    TransCrispr: Transformer Based Hybrid Model for Predicting CRISPR/Cas9
    Single Guide RNA Cleavage Efficiency
    https://github.com/BioinfoApollo/TransCrispr/blob/main/BioNet.py
    """
    def __init__(self, params):
        super(transformer_ont_biofeat_classification, self).__init__()

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

        self.trans_ori_pos = Predict_encoder(
            nhead=params['num_head1'],
            layers=params['transformer_num_layers1'],
            hidden_dim=params['hidden_dim1'],
            latent_dim=params['latent_dim1'],
            embedding_dim=params['nuc_embedding_outputdim'],
            seq_len=params['conv1d_filters_num'],
            probs=params['dropout_rate'],
            device='cuda'
        )
        self.trans_dim_pos = Predict_encoder(
            nhead=params['num_head2'],
            layers=params['transformer_num_layers2'],
            hidden_dim=params['hidden_dim2'],
            latent_dim=params['latent_dim2'],
            embedding_dim=params['nuc_embedding_outputdim'],
            seq_len=params['conv1d_filters_num'],
            probs=params['dropout_rate'],
            device='cuda'
        )

        self.embedding_ori = torch.nn.Embedding(50, params['nuc_embedding_outputdim'])
        self.embedding_dim = torch.nn.Embedding(50, params['nuc_embedding_outputdim'])
        self.embedding_pos = torch.nn.Embedding(50, params['nuc_embedding_outputdim'])

        self.cnov1d_ori = nn.Conv1d(
            params['seq_len'],
            params['conv1d_filters_num'],
            kernel_size=2 * params['conv1d_filters_size'] + 1,
            padding=params['conv1d_filters_size']
        )
        self.cnov1d_dim = nn.Conv1d(
            params['seq_len'],
            params['conv1d_filters_num'],
            kernel_size=2 * params['conv1d_filters_size'] + 1,
            padding=params['conv1d_filters_size']
        )
        self.cnov1d_pos = nn.Conv1d(
            params['seq_len'],
            params['conv1d_filters_num'],
            kernel_size=2 * params['conv1d_filters_size'] + 1,
            padding=params['conv1d_filters_size']
        )

        self.conv2 = nn.Conv1d(
            params['conv1d_filters_num'],
            params['conv1d_filters_num'],
            kernel_size=2 * params['conv1d_filters_size'] + 1,
            padding=params['conv1d_filters_size']
        )
        self.conv3 = nn.Conv1d(
            params['conv1d_filters_num'],
            params['conv1d_filters_num'],
            kernel_size=2 * params['conv1d_filters_size'] + 1,
            padding=params['conv1d_filters_size']
        )

        self.ac = nn.ReLU()

        self.dropout_ori = nn.Dropout(p=params['dropout_rate'])
        self.dropout_dim = nn.Dropout(p=params['dropout_rate'])
        self.dropout_fc = nn.Dropout(p=params['dropout_rate'])

        self.final_fc1 = nn.Linear(
            params['latent_dim1'] + params['latent_dim2'] + 4 + params['nuc_embedding_outputdim'],
            params['fc_hidden1']
        )
        self.final_fcbn1 = nn.BatchNorm1d(params['fc_hidden1'])

        self.final_fc2 = nn.Linear(params['fc_hidden1'], params['fc_hidden2'])
        self.final_fcbn2 = nn.BatchNorm1d(params['fc_hidden2'])

        self.final_fc3 = nn.Linear(params['fc_hidden2'], params['fc_hidden3'])
        self.final_fcbn3 = nn.BatchNorm1d(params['fc_hidden3'])

        self.final_fc4 = nn.Linear(params['fc_hidden3'], 5)

    def forward(self, x, bio):
        input_ori = x[:, 0, :]
        input_dim = x[:, 1, :]
        input_pos = x[:, 2, :] - 1

        embeded_ori = self.embedding_ori(input_ori)
        embeded_dim = self.embedding_dim(input_dim)
        embeded_pos = self.embedding_pos(input_pos)

        conv1_nuc = self.cnov1d_ori(embeded_ori)
        conv1_dimer = self.cnov1d_dim(embeded_dim)

        conv1_nuc = self.ac(conv1_nuc)
        conv1_dimer = self.ac(conv1_dimer)

        pool1_nuc = torch.mean(conv1_nuc, dim=1)
        pool1_dimer = torch.mean(conv1_dimer, dim=1)

        drop1_dimer = self.dropout_dim(conv1_dimer)
        drop1_nuc = self.dropout_dim(conv1_nuc)

        drop1_ori = drop1_nuc + self.cnov1d_pos(embeded_pos)
        drop1_dim = drop1_dimer + self.cnov1d_pos(embeded_pos)

        conv2 = self.conv2(drop1_ori)
        conv3 = self.conv3(drop1_dim)

        conv2 = self.ac(conv2)
        conv3 = self.ac(conv3)

        ori_pos = self.trans_ori_pos(conv2)
        dim_pos = self.trans_dim_pos(conv3)

        output = torch.cat((0.2 * (pool1_nuc + pool1_dimer), 0.8 * ori_pos, 0.8 * dim_pos, bio), dim=-1)

        output = self.final_fc1(output)
        output = self.ac(output)
        output = self.final_fcbn1(output)

        output = self.final_fc2(output)
        output = self.ac(output)
        output = self.final_fcbn2(output)

        output = self.final_fc3(output)
        output = self.ac(output)
        output = self.final_fcbn3(output)

        output = self.final_fc4(output)

        return output


class transformer_ont_biofeat(torch.nn.Module):
    """
    Transformer + CNN + biological features regression model.
    Inspired by:
    TransCrispr: Transformer Based Hybrid Model for Predicting CRISPR/Cas9
    Single Guide RNA Cleavage Efficiency
    https://github.com/BioinfoApollo/TransCrispr/blob/main/BioNet.py
    """
    def __init__(self, params):
        super(transformer_ont_biofeat, self).__init__()

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

        self.trans_ori_pos = Predict_encoder(
            nhead=params['num_head1'],
            layers=params['transformer_num_layers1'],
            hidden_dim=params['hidden_dim1'],
            latent_dim=params['latent_dim1'],
            embedding_dim=params['nuc_embedding_outputdim'],
            seq_len=params['conv1d_filters_num'],
            probs=0.1,
            device='cuda'
        )
        self.trans_dim_pos = Predict_encoder(
            nhead=params['num_head2'],
            layers=params['transformer_num_layers2'],
            hidden_dim=params['hidden_dim2'],
            latent_dim=params['latent_dim2'],
            embedding_dim=params['nuc_embedding_outputdim'],
            seq_len=params['conv1d_filters_num'],
            probs=0.1,
            device='cuda'
        )

        self.embedding_ori = torch.nn.Embedding(50, params['nuc_embedding_outputdim'])
        self.embedding_dim = torch.nn.Embedding(50, params['nuc_embedding_outputdim'])
        self.embedding_pos = torch.nn.Embedding(50, params['nuc_embedding_outputdim'])

        self.cnov1d_ori = nn.Conv1d(
            params['seq_len'],
            params['conv1d_filters_num'],
            kernel_size=2 * params['conv1d_filters_size'] + 1,
            padding=params['conv1d_filters_size']
        )
        self.cnov1d_dim = nn.Conv1d(
            params['seq_len'],
            params['conv1d_filters_num'],
            kernel_size=2 * params['conv1d_filters_size'] + 1,
            padding=params['conv1d_filters_size']
        )
        self.cnov1d_pos = nn.Conv1d(
            params['seq_len'],
            params['conv1d_filters_num'],
            kernel_size=2 * params['conv1d_filters_size'] + 1,
            padding=params['conv1d_filters_size']
        )

        self.conv2 = nn.Conv1d(
            params['conv1d_filters_num'],
            params['conv1d_filters_num'],
            kernel_size=2 * params['conv1d_filters_size'] + 1,
            padding=params['conv1d_filters_size']
        )
        self.conv3 = nn.Conv1d(
            params['conv1d_filters_num'],
            params['conv1d_filters_num'],
            kernel_size=2 * params['conv1d_filters_size'] + 1,
            padding=params['conv1d_filters_size']
        )

        self.ac = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        self.dropout_ori = nn.Dropout(p=params['dropout_rate'])
        self.dropout_dim = nn.Dropout(p=params['dropout_rate'])
        self.dropout_fc = nn.Dropout(p=params['dropout_rate'])

        self.final_fc1 = nn.Linear(
            params['latent_dim1'] + params['latent_dim2'] + 4 + params['nuc_embedding_outputdim'],
            params['fc_hidden1']
        )
        self.final_fc2 = nn.Linear(params['fc_hidden1'], params['fc_hidden2'])
        self.final_fc3 = nn.Linear(params['fc_hidden2'], params['fc_hidden3'])
        self.final_fc4 = nn.Linear(params['fc_hidden3'], 1)

    def forward(self, x, bio):
        input_ori = x[:, 0, :]
        input_dim = x[:, 1, :]
        input_pos = x[:, 2, :] - 1

        embeded_ori = self.embedding_ori(input_ori)
        embeded_dim = self.embedding_dim(input_dim)
        embeded_pos = self.embedding_pos(input_pos)

        conv1_nuc = self.cnov1d_ori(embeded_ori)
        conv1_dimer = self.cnov1d_dim(embeded_dim)

        conv1_nuc = self.ac(conv1_nuc)
        conv1_dimer = self.ac(conv1_dimer)

        pool1_nuc = torch.mean(conv1_nuc, dim=1)
        pool1_dimer = torch.mean(conv1_dimer, dim=1)

        drop1_dimer = self.dropout_dim(conv1_dimer)
        drop1_nuc = self.dropout_dim(conv1_nuc)

        drop1_ori = drop1_nuc + self.cnov1d_pos(embeded_pos)
        drop1_dim = drop1_dimer + self.cnov1d_pos(embeded_pos)

        conv2 = self.conv2(drop1_ori)
        conv3 = self.conv3(drop1_dim)

        conv2 = self.ac(conv2)
        conv3 = self.ac(conv3)

        ori_pos = self.trans_ori_pos(conv2)
        dim_pos = self.trans_dim_pos(conv3)

        output = torch.cat((0.2 * (pool1_nuc + pool1_dimer), 0.8 * ori_pos, 0.8 * dim_pos, bio), dim=-1)

        output = self.final_fc1(output)
        output = self.ac(output)
        output = self.dropout_fc(output)

        output = self.final_fc2(output)
        output = self.ac(output)
        output = self.dropout_fc(output)

        output = self.final_fc3(output)
        output = self.ac(output)
        output = self.dropout_fc(output)

        output = self.final_fc4(output)

        return self.sigmoid(output)


class Predict_transformer_bio(torch.nn.Module):
    def __init__(self, params):
        super(Predict_transformer_bio, self).__init__()

        self.dropout_rate_fc = params['dropout_rate_fc']
        self.relu = nn.ReLU()

        self.trans_ori_pos = Predict_encoder(
            nhead=params['num_head1'],
            layers=params['transformer_num_layers1'],
            hidden_dim=params['hidden_dim1'],
            latent_dim=params['latent_dim1'],
            embedding_dim=params['embedding_dim1'],
            seq_len=params['seq_len'],
            probs=params['dropout_rate1'],
            device='cuda'
        )
        self.trans_dim_pos = Predict_encoder(
            nhead=params['num_head2'],
            layers=params['transformer_num_layers2'],
            hidden_dim=params['hidden_dim2'],
            latent_dim=params['latent_dim2'],
            embedding_dim=params['embedding_dim2'],
            seq_len=params['seq_len'],
            probs=params['dropout_rate2'],
            device='cuda'
        )

        self.embedding_ori = torch.nn.Embedding(100, params['embedding_dim1'])
        self.embedding_dim = torch.nn.Embedding(100, params['embedding_dim2'])

        self.ac = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=self.dropout_rate_fc)

        self.final_fc1 = nn.Linear(params['latent_dim1'] + params['latent_dim2'] + 4, params['fc_hidden1'])
        self.final_fc2 = nn.Linear(params['fc_hidden1'], params['fc_hidden2'])
        self.final_fc3 = nn.Linear(params['fc_hidden2'], 1)

    def forward(self, x, bio):
        input_ori = x[:, 0, :]
        input_dim = x[:, 1, :]

        embeded_ori = self.embedding_ori(input_ori)
        embeded_dim = self.embedding_dim(input_dim)

        ori_pos = self.trans_ori_pos(embeded_ori)
        dim_pos = self.trans_dim_pos(embeded_dim)

        output = torch.cat((ori_pos, dim_pos, bio), dim=-1)

        output = self.final_fc1(output)
        output = self.ac(output)
        output = self.dropout(output)

        output = self.final_fc2(output)
        output = self.ac(output)
        output = self.dropout(output)

        output = self.final_fc3(output)
        output = self.ac(output)

        return self.relu(output)


class Classification_transformer(torch.nn.Module):
    def __init__(self, params):
        super(Classification_transformer, self).__init__()

        self.dropout_rate_fc = params['dropout_rate_fc']
        self.relu = nn.ReLU()

        self.trans_ori_pos = Predict_encoder(
            nhead=params['num_head1'],
            layers=params['transformer_num_layers1'],
            hidden_dim=params['hidden_dim1'],
            latent_dim=params['latent_dim1'],
            embedding_dim=params['embedding_dim1'],
            seq_len=params['seq_len'],
            probs=params['dropout_rate1'],
            device='cuda'
        )
        self.trans_dim_pos = Predict_encoder(
            nhead=params['num_head2'],
            layers=params['transformer_num_layers2'],
            hidden_dim=params['hidden_dim2'],
            latent_dim=params['latent_dim2'],
            embedding_dim=params['embedding_dim2'],
            seq_len=params['seq_len'],
            probs=params['dropout_rate2'],
            device='cuda'
        )

        self.embedding_ori = torch.nn.Embedding(100, params['embedding_dim1'])
        self.embedding_dim = torch.nn.Embedding(100, params['embedding_dim2'])

        self.ac = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=self.dropout_rate_fc)

        self.final_fc1 = nn.Linear(params['latent_dim1'] + params['latent_dim2'], params['fc_hidden1'])
        self.final_fc2 = nn.Linear(params['fc_hidden1'], params['fc_hidden2'])
        self.final_fc3 = nn.Linear(params['fc_hidden2'], 5)

    def forward(self, X):
        x = X.to(torch.int)

        input_ori = x[:, 0, :]
        input_dim = x[:, 1, :]

        embeded_ori = self.embedding_ori(input_ori)
        embeded_dim = self.embedding_dim(input_dim)

        ori_pos = self.trans_ori_pos(embeded_ori)
        dim_pos = self.trans_dim_pos(embeded_dim)

        output = torch.cat((ori_pos, dim_pos), dim=-1)

        output = self.final_fc1(output)
        output = self.ac(output)
        output = self.dropout(output)

        output = self.final_fc2(output)
        output = self.ac(output)

        output = self.final_fc3(output)

        return self.relu(output)


class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_num, hidden_layer_units_num, dropout):
        super(MLP, self).__init__()
        layers = []
        layers.append(torch.nn.Linear(input_dim, hidden_layer_units_num))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout))
        for _ in range(hidden_layer_num - 1):
            layers.append(torch.nn.Linear(hidden_layer_units_num, hidden_layer_units_num))
            layers.append(torch.nn.BatchNorm1d(hidden_layer_units_num))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(hidden_layer_units_num, output_dim))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


if __name__ == '__main__':
    params = {
        'train_batch_size': 64,
        'train_epochs_num': 100,
        'train_base_learning_rate': 0.00005,
        'model_save_file': './models/BestModel_WT_withbio.h5',
        'dropout_rate': 0.2,
        'nuc_embedding_outputdim': 100,
        'conv1d_filters_size': 7,
        'conv1d_filters_num': 512,
        'transformer_num_layers': 4,
        'transformer_final_fn': 198,
        'transformer_ffn_1stlayer': 111,
        'dense1': 176,
        'dense2': 88,
        'dense3': 22
    }
    in_channles = 3
    x = torch.ones(size=(64, in_channles, 100), device='cuda')
    predict_model = Predict_transformer(params).to('cuda')
    print(predict_model(x).shape)
