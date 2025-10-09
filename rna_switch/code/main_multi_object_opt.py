import os
import time
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
import random



def dna_reverse_complement_to_rna(dna_seq: str) -> str:
    """
    将DNA序列反向互补后转为RNA（T->U）
    """
    dna_seq = dna_seq.upper()
    rc_seq = str(Seq(dna_seq).reverse_complement())  # 得到反向互补
    rna_seq = rc_seq.replace("T", "U")               # T 转换为 U
    return rna_seq

def construct_toehold_from_list(merged_list):
    """
    输入一个合并序列的列表，每个元素为45nt：switch(30) + stem1(6) + stem2(9)，
    返回完整toehold switch序列列表。

    参数:
    - merged_list: List[str]，每个元素是一个45nt的合并序列

    返回:
    - List[str]，每个是拼接后的完整 toehold switch 序列
    """
    # 固定结构
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
    读取FASTA文件，预测其RNA二级结构并按自由能排序，保存为CSV。
    
    参数:
        fasta_file (str): 输入FASTA文件名
        output_csv (str): 结果保存的CSV文件名
    """

    toehold_switch_sequence, Trigger_rnas = construct_toehold_from_list(merged_list=switch)

    # 存储序列结构信息
    MFE_selfs = []
    DeltaDeltaG_opens=[]
    records = []



    for seq, trigger in zip(toehold_switch_sequence, Trigger_rnas):

        structure, mfe = RNA.fold(seq)
        res =  analyze_rna_switch(switch_seq=seq, trigger_seq=trigger)
        DeltaDeltaG_opens.append(round(res["DeltaDeltaG_open"], 4))

        records.append({

            "Sequence": seq,
            "Trigger rna": trigger,
            "Structure": structure,

            "MFE_self (kcal/mol)": round(res["MFE_self"], 2),
            # "MFE (kcal/mol)": round(mfe, 2),
            "MFE_hybrid (kcal/mol)": round(res["MFE_hybrid"], 2),
            "DeltaDeltaG_open (kcal/mol)": round(res["DeltaDeltaG_open"], 2),
            
        })

    # 排序并导出为 CSV
    # df = pd.DataFrame(records)
    # df_sorted = df.sort_values(by="DeltaDeltaG_open (kcal/mol)", ascending=True)

    return toehold_switch_sequence, Trigger_rnas, DeltaDeltaG_opens
    # df_sorted.to_csv(output_csv, index=False)



def construct_toehold_from_list(merged_list):
    """
    输入一个合并序列的列表，每个元素为45nt：switch(30) + stem1(6) + stem2(9)，
    返回完整toehold switch序列列表。

    参数:
    - merged_list: List[str]，每个元素是一个45nt的合并序列

    返回:
    - List[str]，每个是拼接后的完整 toehold switch 序列
    """
    # 固定结构
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
    根据 RNA 序列构建结构数组（N × N），每个位置表示两碱基的配对潜力（以氢键数量为值）

    参数:
        seq (str): 输入 RNA 序列（如 'AUGCGAU...'）

    返回:
        np.ndarray: 大小为 N×N 的结构数组，按配对规则编码为 0, 2, 3
    """
    seq = seq.upper().replace('T', 'U')  # 兼容 DNA 序列输入
    N = len(seq)
    struct_array = np.zeros((N, N), dtype=int)

    # 合法配对及其对应氢键数
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

    # 先整合成完整的mRNA序列，然后进行编码，为预测模型提供输入
    seqList = construct_toehold_from_list(merged_list=seqList)

    structures=np.array([build_structure_array(seq=mRNA) for mRNA in seqList])
    X_seq=np.array([Dimer_split_seqs(sequence) for sequence in seqList])

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


    '''
    开始预测表达量
    '''

    Dimer, struc = encode_structure_list(sequences)
    Dimer = torch.tensor(Dimer).to(device)
    struc = torch.tensor(struc).to(device)

    print(Dimer.shape)
    print(struc.shape)

    ons=compute_scaler(prediction_on(Dimer, struc)) #Computes predictions for the random set
    offs=compute_scaler(prediction_off(Dimer, struc)) #Computes predictions for the random set

    ons = np.array(ons)
    ons = ons.squeeze()

    offs = np.array(offs)
    offs = offs.squeeze()

    print('offs = ', offs)
    

    if opt:
        all_sequences, all_ons, all_offs, all_radios = optimize_tensor_input_with_cmaes(diffusion_model = diffusion, predictor_on = prediction_on, predictor_off = prediction_off, number=promoters_number, 
                                     sigma=0.5, max_iter=30, popsize=promoters_number)
        
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

        if random.random() < 0.6242:
            sequences.append("A" + decoded_sequence)
        else:
            sequences.append("C" + decoded_sequence)
        # sequences.append("A" + decoded_sequence)


    '''
    计算能量
    '''
    # 先
    toehold_switch_sequence, Trigger_rnas, DeltaDeltaG_opens = process_toehold_structures(switch=sequences)


    '''
    开始预测表达量
    '''

    Dimer, struc = encode_structure_list(sequences)
    Dimer = torch.tensor(Dimer).to(device)
    struc = torch.tensor(struc).to(device)

    print(Dimer.shape)
    print(struc.shape)

    ons=compute_scaler(predictor_on(Dimer, struc)) #Computes predictions for the random set
    offs=compute_scaler(predictor_off(Dimer, struc)) #Computes predictions for the random set

    ons = np.array(ons)
    ons = ons.squeeze()
    # ons = ons.tolist()

    offs = np.array(offs)
    offs = offs.squeeze()
    # offs = offs.tolist()

    radios = ons - offs
    mean_radio = np.nanmean(radios)  # 忽略 NaN 取平均

    return -mean_radio, sequences, ons, offs, radios, toehold_switch_sequence, Trigger_rnas, DeltaDeltaG_opens # 仍然是最小化问题


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
    输入:
        prediction_model: 训练好的预测模型（输入: 编码序列 → 输出: raw score）
        promoter: list of promoter sequences (str)

    输出:
        avg_strength: 经指数转换后的平均表达强度（float）
    """
    # 1. 序列特征编码
    features = [np.array(Dimer_split_seqs(seq)) for seq in promoter]
    features = np.array(features)  # shape: (N, F)

    # 2. 转为 Tensor
    encoded_sequences = torch.tensor(features, dtype=torch.float32).to(device)

    # 3. 模型预测 raw 分数
    raw_scores = prediction_model(encoded_sequences)  # shape: (N,)
    
    # 4. 归一化并指数转换
    min_strength = -8.6382
    max_strength = 12.5883
    raw_scores = raw_scores.squeeze().cpu().numpy()  # 确保是 NumPy 数组

    transformed_scores = 2 ** (raw_scores * (max_strength - min_strength) + min_strength)

    # 5. 取平均
    avg_strength = transformed_scores.mean()
    print('avg_strength = ',avg_strength)

    return avg_strength


def optimize_tensor_input_with_cmaes(diffusion_model, predictor_on, predictor_off, number=512, 
                                     sigma=0.5, max_iter=5, popsize=512):
    """
    使用 CMA-ES 优化一个 tensor 输入（如 soft one-hot 表示）以最大化预测强度
    
    shape: 优化张量的形状，例：(64, 4, 44)
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

            score, sequences, ons, offs, radios, toehold_switch_sequence, Trigger_rnas, DeltaDeltaG_opens  = blackbox_objective_z_tensor(s, diffusion_model, predictor_on, predictor_off, shape=shape)

            df = pd.DataFrame({
                'on': ons,
                'off': offs,
                'radio': radios,
                'sequence': sequences,
                'toehold_switch_sequence': toehold_switch_sequence,
                'Trigger_rnas': Trigger_rnas,
                'DeltaDeltaG_opens': DeltaDeltaG_opens,
                })
            
            df.to_csv(f'../result/output_opt_iter={gen}_score={score}.csv', index=False)

            
            all_sequences += sequences
            all_offs += offs.tolist()
            all_ons += ons.tolist()
            all_radios += radios.tolist()

            
            scores.append(score)

            # 记录最优
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
            result.append(float('inf'))  # 或者用 0, None, np.nan 等处理方式
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
         
        # 按 strength 降序排序，并保留前 100 行
        df_sorted = df.sort_values(by='radio', ascending=False).head(promoter_number*3)
        df_sorted.to_csv('../result/output.csv', index=False)

    

    
    