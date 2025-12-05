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
    def __init__(self, num_channels,kernel_size,padding):
        super(ResidualBlock, self).__init__()
        self.num_channels = num_channels

        # define two convolutional layers
        self.conv1 = nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, padding=padding)
        self.batch_norm = nn.BatchNorm1d(num_channels) 

        self.ac = nn.LeakyReLU()
        
    def forward(self, x):
        res = x
        for _ in range(2):
            res = self.conv1(res)
            res = self.batch_norm(res) 
            # res = F.relu(res)
            res = self.ac(res)

            res = self.conv2(res)
            res = self.batch_norm(res)
            # res = F.relu(res)
            
        return x + res


class Predict_translation_off(torch.nn.Module):
    
    def __init__(self,params):
        super(Predict_translation_off, self).__init__()

        self.dropout_rate_fc = params['dropout_rate_fc']
        self.relu = nn.ReLU()

        self.trans_ori_pos = Predict_encoder(nhead = params['num_head1'],layers = params['transformer_num_layers1'],hidden_dim=params['hidden_dim1'],latent_dim=params['latent_dim1'],embedding_dim=params['embedding_dim1'],seq_len=params['seq_len'],probs=params['dropout_rate1'],device='cuda')
        self.trans_dim_pos = Predict_encoder(nhead = params['num_head2'],layers = params['transformer_num_layers2'],hidden_dim=params['hidden_dim2'],latent_dim=params['latent_dim2'],embedding_dim=params['embedding_dim2'],seq_len=params['seq_len'],probs=params['dropout_rate2'],device='cuda')

        self.embedding_ori = torch.nn.Embedding(100, params['embedding_dim1'])
        self.embedding_dim = torch.nn.Embedding(100, params['embedding_dim2'])

        self.ac = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=self.dropout_rate_fc)
        
        self.final_fc1 = nn.Linear(params['latent_dim1'] + params['latent_dim2'], params['fc_hidden1']) # 3+3+2+1=9 reserved for additional biological features
        self.final_fc2 = nn.Linear(params['fc_hidden1'],params['fc_hidden2'])
        self.final_fc3 = nn.Linear(params['fc_hidden2'],1)

        
    def forward(self, X):

        x = X.to(torch.int)
 
        input_ori = x[:, 0, :]
        input_dim = x[:, 1, :]
        # input_pos = x[:, 2, :] - 1
        # print('input_ori.shape = ',input_ori.shape)
        # print('input_dim.shape = ',input_dim.shape)
        # print('input_pos.shape = ',input_pos.shape)
        # print('input_ori = ',input_ori)
        
        # print('start embedding')
        embeded_ori = self.embedding_ori(input_ori)
        embeded_dim = self.embedding_dim(input_dim)

        
        ori_pos = self.trans_ori_pos(embeded_ori)
        dim_pos = self.trans_dim_pos(embeded_dim)
        # print('end transformer encoder')
        
        output = torch.cat((ori_pos, dim_pos), dim=-1) # concatenate transformer outputs (and optionally biological features)
        # output = self.mlp(ori_dim_pos)
        
        output = self.final_fc1(output)
        output = self.ac(output)
        # output = self.relu(output)
        output = self.dropout(output)

        output = self.final_fc2(output)
        output = self.ac(output)
        # output = self.dropout(output)

        output = self.final_fc3(output)
        
        # output = self.relu(output)
        # print('output.shape', output.shape)
        # pdb.set_trace()
        return self.relu(output)


class Predict_translation(torch.nn.Module):
    
    def __init__(self,params):
        super(Predict_translation, self).__init__()

        self.dropout_rate_fc = params['dropout_rate_fc']
        self.relu = nn.ReLU()

        self.trans_ori_pos = Predict_encoder(nhead = params['num_head1'],layers = params['transformer_num_layers1'],hidden_dim=params['hidden_dim1'],latent_dim=params['latent_dim1'],embedding_dim=params['embedding_dim1'],seq_len=params['seq_len'],probs=params['dropout_rate1'],device='cuda')
        self.trans_dim_pos = Predict_encoder(nhead = params['num_head2'],layers = params['transformer_num_layers2'],hidden_dim=params['hidden_dim2'],latent_dim=params['latent_dim2'],embedding_dim=params['embedding_dim2'],seq_len=params['seq_len'],probs=params['dropout_rate2'],device='cuda')

        self.embedding_ori = torch.nn.Embedding(100, params['embedding_dim1'])
        self.embedding_dim = torch.nn.Embedding(100, params['embedding_dim2'])
        
        # dropout layer
        self.ac = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=self.dropout_rate_fc)
        
        self.final_fc1 = nn.Linear(params['latent_dim1'] + params['latent_dim2'], params['fc_hidden1']) # 3+3+2+1=9 reserved for additional biological features
        self.final_fc2 = nn.Linear(params['fc_hidden1'],params['fc_hidden2'])
        self.final_fc3 = nn.Linear(params['fc_hidden2'],1)
        # self.bio_fc1 = nn.Linear(13, params['fc_hidden1'])
        
    def forward(self, X):

        x = X.to(torch.int)

        input_ori = x[:, 0, :]
        input_dim = x[:, 1, :]

        embeded_ori = self.embedding_ori(input_ori)
        embeded_dim = self.embedding_dim(input_dim)
        
        ori_pos = self.trans_ori_pos(embeded_ori)
        dim_pos = self.trans_dim_pos(embeded_dim)

        
        output = torch.cat((ori_pos, dim_pos), dim=-1) # concatenate transformer outputs (and optionally biological features)
        
        output = self.final_fc1(output)
        output = self.ac(output)

        output = self.dropout(output)

        output = self.final_fc2(output)
        output = self.ac(output)
 

        output = self.final_fc3(output)

        return self.relu(output)




class CNN1D_Flatten(nn.Module):
    
    def __init__(self, fc_hidden=512):
        super(CNN1D_Flatten, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=115, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.pool1 = nn.MaxPool1d(kernel_size=5, stride=5)  # output length 23

        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=5, stride=5)  # output length 4

        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)  # output length remains 4

        # 512 × 4 = 2048
        self.fc = nn.Linear(2048, fc_hidden)
        
    def forward(self, x):
        
        # print(x.shape)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        # print(x.shape)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # print(x.shape)
        x = F.relu(self.conv3(x))  # output shape [B, 256, L]
        # print(x.shape)
        x = x.flatten(start_dim=1)  # flatten to [B, 256 * L]
        # print(x.shape)
        x = self.fc(x)              # map to [B, fc_hidden]

        return x


class CrossAttentionFusionSameDim(nn.Module):
    
    def __init__(self, dim=256, num_heads=4, dropout=0.1):
        super().__init__()

        self.attn1 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn2 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn3 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.fusion = nn.Linear(dim * 3, dim)  # fuse three attention outputs

    def forward(self, x1, x2, x3):
        # input: [B, D] → [B, 1, D]
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        x3 = x3.unsqueeze(1)

        # cross attention: each vector attends another
        o1, _ = self.attn1(query=x1, key=x2, value=x2)  # x1 attends x2
        o2, _ = self.attn2(query=x2, key=x3, value=x3)  # x2 attends x3
        o3, _ = self.attn3(query=x3, key=x1, value=x1)  # x3 attends x1

        # concatenate outputs and project back
        fused = torch.cat([o1, o2, o3], dim=-1)  # [B, 1, 3*D]
        fused = self.fusion(fused)              # [B, 1, D]
        return fused.squeeze(1)                 # [B, D]


class Predict_translation_structure(torch.nn.Module): # sequence expert + dimer expert + secondary structure expert
    
    def __init__(self,params):
        super(Predict_translation_structure, self).__init__()

        self.dropout_rate_fc = params['dropout_rate_fc']
        self.relu = nn.ReLU()
        self.cnn = CNN1D_Flatten(fc_hidden=params['latent_dim'])

        self.trans_ori_pos = Predict_encoder(nhead = params['num_head1'],layers = params['transformer_num_layers1'],hidden_dim=params['hidden_dim1'],latent_dim=params['latent_dim'],embedding_dim=params['embedding_dim1'],seq_len=params['seq_len'],probs=params['dropout_rate1'],device='cuda')
        self.trans_dim_pos = Predict_encoder(nhead = params['num_head2'],layers = params['transformer_num_layers2'],hidden_dim=params['hidden_dim2'],latent_dim=params['latent_dim'],embedding_dim=params['embedding_dim2'],seq_len=params['seq_len'],probs=params['dropout_rate2'],device='cuda')

        self.embedding_ori = torch.nn.Embedding(100, params['embedding_dim1'])
        self.embedding_dim = torch.nn.Embedding(100, params['embedding_dim2'])
        
        # dropout layer
        self.ac = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=self.dropout_rate_fc)
        self.CrossAttention = CrossAttentionFusionSameDim(dim=params['latent_dim'])
        
        self.final_fc1 = nn.Linear(params['latent_dim'], params['fc_hidden1']) # 3+3+2+1=9 reserved for additional biological features
        self.final_fc2 = nn.Linear(params['fc_hidden1'],params['fc_hidden2'])
        self.final_fc3 = nn.Linear(params['fc_hidden2'],1)
 
        
    def forward(self, X, structure):

        x = X.to(torch.int)
        
        structure = self.cnn(structure.to(torch.float))

        input_ori = x[:, 0, :]
        input_dim = x[:, 1, :]

        embeded_ori = self.embedding_ori(input_ori)
        embeded_dim = self.embedding_dim(input_dim)
        
        ori_pos = self.trans_ori_pos(embeded_ori)
        dim_pos = self.trans_dim_pos(embeded_dim)

        
        # output = torch.cat((ori_pos, dim_pos), dim=-1) # concatenate transformer outputs (and optionally biological features)
        
        output = self.CrossAttention(dim_pos, ori_pos, structure)
        
        output = self.final_fc1(output)
        output = self.ac(output)

        output = self.dropout(output)

        output = self.final_fc2(output)
        output = self.ac(output)
 

        output = self.final_fc3(output)

        return self.relu(output)
    
    
    
class CNN_OneHot_Seq(nn.Module):
    def __init__(self, input_length=115, task_type='regression'):
        super(CNN_OneHot_Seq, self).__init__()

        # input channels = 4 (one-hot encoding of A, C, G, U)
        self.conv1 = nn.Conv1d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)

        # flattened size: 128 × input_length
        self.flattened_dim = 128 * input_length
        self.fc1 = nn.Linear(self.flattened_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.bn_fc1 = nn.BatchNorm1d(16)
        self.bn_fc2 = nn.BatchNorm1d(16)

        if task_type == 'regression':
            self.out = nn.Linear(16, 1)  # ON, OFF, ON/OFF
        elif task_type == 'classification':
            self.out = nn.Linear(16, 2)  # softmax output
        else:
            raise ValueError("task_type must be 'regression' or 'classification'")

        self.task_type = task_type
        

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)

        x = self.out(x)

        if self.task_type == 'classification':
            x = F.softmax(x, dim=1)

        return x


if __name__ == "__main__":
    
    cnn = CNN1D_Flatten(fc_hidden=128)
    input_tensor = torch.randn(64, 115, 115)
    output_tensor = cnn(input_tensor)
