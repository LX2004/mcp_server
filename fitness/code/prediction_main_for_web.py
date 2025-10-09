import numpy as np
from utils import *
from net import predict_transformerv2

import matplotlib.pyplot as plt
import seaborn as sns

def save_predictions_density(predictions, file_path="predictions_density.png"):
    """
    绘制预测值的概率密度图，并保存为指定路径的文件。

    参数:
        predictions (list): 输入的预测值列表。
        file_path (str): 图像保存的文件路径，默认保存为 'predictions_density.png'。
    """
    # 设置图表大小
    plt.figure(figsize=(8, 6))
    
    # 绘制概率密度曲线
    sns.kdeplot(predictions, fill=True, color='blue')
    
    # 添加标题和坐标轴标签
    plt.title("Probability Density Curve of Predictions")
    plt.xlabel("Prediction Values")
    plt.ylabel("Density")
    
    # 保存图表到文件
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()  # 关闭图表，释放内存

    print(f"图表已保存至: {file_path}")


def read_data(filename):

    guides = []
    essentials = []

    oris = []
    codings = []


    with open(filename, 'r') as file:

        reader = csv.reader(file)
        header = next(reader)  # 读取文件头部
        
        guide_idx = header.index('guide_rna')  # 获取'guide'列的索引
        essential_idx = header.index('essential')  # 获取'essential'列的索引

        ori_idx = header.index('ori')  # 获取'essential'列的索引
        coding_idx = header.index('coding')  # 获取'coding'列的索引

        
        for row in reader:

            guide = row[guide_idx]  # 获取'guide'列的值
            essential = row[essential_idx]  # 获取'essential'列的值

            ori = row[ori_idx]  # 获取'ori'列的值
            coding = row[coding_idx]  # 获取'coding'列的值


            if isinstance(guide, str) and isinstance(essential, str) and isinstance(ori, str):

                guides.append(guide)
                essentials.append(essential)

                oris.append(ori)
                codings.append(coding)

    return guides, essentials, oris, codings
    # return np.array(guides), np.array(fit18s)


def make_dataset_for_find_prometer(guides, essentials, oris, codings):

    bio = []
    seq = []

    for sequence,  essential, ori, coding in zip(guides, essentials, oris, codings):

        split_sequence = [char for char in sequence]
        seq.append(split_sequence)

        # 处理生物信息量
        essential_feature = encode_essential(essential)
        ori_feature = encode_ori(ori)
        coding_feature = encode_coding(coding)

        # 将浮点数添加到堆叠后的数组中
        biofeature = np.concatenate((essential_feature, ori_feature, coding_feature))
        bio.append(biofeature)

    return np.array(seq), np.array(bio)

def compute_scaler(model_output):

    model_output = model_output.detach().cpu().numpy()

    # max_reads =  2.5
    # min_reads =  -3.5

    max_reads =  1.70545308756268
    min_reads =  -3.5

    model_output = min_reads + (max_reads - min_reads) * model_output

    return model_output

def encode_list(seqList):

    X_seq=np.array([Dimer_split_seqs(''.join(np.char.join('', sequence))) for sequence in seqList])

    return X_seq

def plot_target_label(seq, bio, model): # 绘制实际值和预测值的分布
    
    randomset_=encode_list(seq)
    randomset_ = torch.tensor(randomset_).to(device)

    bio_  = torch.tensor(bio).to(device)
    df_pred = compute_scaler(model(randomset_, bio_))

    df_pred = np.array(df_pred).squeeze()
    return df_pred

if __name__ == '__main__':

    # 确定训练所需GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(2)
    print('device =',device)

    # 加载模型
    model = torch.load('../model/prediction_model.pth').to(device)

    # 从数据库中加载
    filename = "../data/example.csv"
    guides, essentials, oris, codings = read_data(filename=filename)
    seq, bio = make_dataset_for_find_prometer(guides, essentials, oris, codings)

    '''计算预测值'''
    predictions = plot_target_label(seq, bio, model)
    predictions = predictions.tolist()

    save_predictions_density(predictions, file_path="predictions_density.png")

    file_path = '../data/example.csv'
    filtered_df = pd.read_csv(file_path)

    filtered_df['prediction_result'] = predictions
    filtered_df = filtered_df.astype({'essential': str, 'ori': str, 'coding': str, 'prediction_result' : float})
    filtered_df['essential'] = filtered_df['essential'].replace({'True': 'TRUE', 'False': 'FALSE'})
    filtered_df['ori'] = filtered_df['ori'].replace({'True': 'TRUE', 'False': 'FALSE'})
    filtered_df['coding'] = filtered_df['coding'].replace({'True': 'TRUE', 'False': 'FALSE'})
    
    # 保存为新的CSV文件
    filtered_df.to_csv('../result/example.csv', index=False, quoting=csv.QUOTE_NONE)








