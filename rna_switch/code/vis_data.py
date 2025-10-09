import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_on_off(csv_file, output_img, eps=1e-6):
    """
    读取CSV文件，计算 ON/OFF 和 ON-OFF，绘制分布图并标注最大值、最小值、平均值
    参数:
        csv_file: str, 输入的CSV文件路径 (必须包含列 'ON' 和 'OFF')
        output_img: str, 输出的图片文件路径
        eps: float, 用于替换 OFF 中的 0，默认 1e-6
    """
    # 读取数据
    df = pd.read_csv(csv_file)

    # 把 OFF 中的 0 替换成一个很小的值
    df["OFF"] = df["OFF"].replace(0, eps)

    # 计算新列
    df["ON_div_OFF"] = df["ON"] / df["OFF"]
    df["ON_minus_OFF"] = df["ON"] - df["OFF"]

    # 设置画布
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cols = ["ON_div_OFF", "ON_minus_OFF"]

    for ax, col in zip(axes, cols):
        data = df[col]

        # 去掉 inf 和 NaN
        data = data.replace([np.inf, -np.inf], np.nan).dropna()

        if data.empty:
            ax.set_title(f"{col}: no valid data")
            continue

        # 画直方图 + KDE
        ax.hist(data, bins=30, alpha=0.6, color="skyblue", density=True, label="Histogram")
        data.plot(kind="kde", ax=ax, color="red", label="KDE")

        # 计算统计值
        max_val = data.max()
        min_val = data.min()
        mean_val = data.mean()

        # 在图上标注
        ax.axvline(max_val, color="green", linestyle="--", label=f"Max={max_val:.2f}")
        ax.axvline(min_val, color="blue", linestyle="--", label=f"Min={min_val:.2f}")
        ax.axvline(mean_val, color="orange", linestyle="--", label=f"Mean={mean_val:.2f}")

        ax.set_title(f"Distribution of {col}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    plt.close()



def read_data(filename):
    """
    读取 CSV 文件，提取 mRNA 结构信息，并返回拼接后的序列以及单独的各个 mRNA 组成部分。
    """

    # 读取 CSV 文件
    df = pd.read_csv(filename)

    # 检查数据列是否存在，防止 KeyError
    required_columns = ['loop1', 'switch', 'loop2', 'stem1', 'atg', 'stem2', 'linker', 'post_linker', 'ON', 'OFF', 'ON_OFF']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Error: 数据集中缺少列 '{col}'")

    # 去除包含 NaN 的行，确保数据完整
    df = df.dropna(subset=['ON', 'OFF', 'ON_OFF'])

    # 转换 ON/OFF 数据类型
    ons = df['ON'].astype(float).tolist()
    offs = df['OFF'].astype(float).tolist()
    on_offs = df['ON_OFF'].astype(float).tolist()

    # 分别获取 mRNA 的各个组成部分
    loop1s = df['loop1'].tolist()
    switches = df['switch'].tolist()
    loop2s = df['loop2'].tolist()
    stem1s = df['stem1'].tolist()
    atgs = df['atg'].tolist()
    stem2s = df['stem2'].tolist()
    linkers = df['linker'].tolist()
    post_linkers = df['post_linker'].tolist()

    # 组合所有部分形成完整的 mRNA 序列
    mRNAs = [l1 + sw + l2 + s1 + atg + s2 + lk + pl for l1, sw, l2, s1, atg, s2, lk, pl in 
             zip(loop1s, switches, loop2s, stem1s, atgs, stem2s, linkers, post_linkers)]
    
    # 分离 mRNA 恒定部分和可变部分
    constant_part = [l1 + atg + lk + pl for l1, atg, lk, pl in zip(loop1s, atgs, linkers, post_linkers)]
    variable_part = [sw + s1 + s2 for sw, s1, s2 in zip(switches, stem1s, stem2s)]
    

    print(f'✅ 读取完成，共 {len(mRNAs)} 条数据')

    return mRNAs, constant_part, variable_part, loop1s, switches, loop2s, stem1s, atgs, stem2s, linkers, post_linkers, ons, offs, on_offs

def check_consistency(*sequences):
    """
    检查传入的多个子序列列表中的所有元素是否完全一致。
    如果某个列表中的所有元素相同，则返回该唯一值，否则返回 None。
    
    参数:
    *sequences - 变长参数，每个参数是一个子序列列表
    
    返回:
    result - 一个字典，键是列表名称，值是该列表的一致值（如果所有元素都相同），否则返回 None
    """
    result = {}
    
    for i, seq_list in enumerate(sequences):
        unique_values = set(seq_list)  # 获取列表中的唯一值集合
        
        if len(unique_values) == 1:  # 如果只有一个唯一值，则所有元素相同
            result[f"sequence_{i+1}"] = list(unique_values)[0]
        else:
            result[f"sequence_{i+1}"] = None  # 存在多个不同值，返回 None
            
    return result

def save_mrna_to_fasta(mrna_list, filename):
    """
    将 mRNA 序列从列表存储到 FASTA 文件。
    
    参数:
    mrna_list: 包含 mRNA 序列的列表
    filename: 要保存的文件名
    """
    with open(filename, 'w') as file:
        for i, mrna in enumerate(mrna_list):
            # 写入标题行
            file.write(f">sequence_{i+1}\n")
            file.write(mrna + "\n")

    print(f"FASTA 文件已保存到 {filename}")


# train(params,train_dataset,test_dataset)
if __name__ == '__main__':
 
    # 处理数据
    filename = '/home/leixin/mcp_server/switch/data/Toehold_mRNA_Dataset_cleanplus.csv'
    analyze_on_off(csv_file=filename, output_img='../data_figure/')


    filename = '/home/liangce/lx/Promoter_mRNA_synthetic/data/Toehold_mRNA_Dataset_clean.csv'
    mRNAs, constant_part, variable_part, loop1s, switches, loop2s, stem1s, atgs, stem2s, linkers, post_linkers, ons, offs, on_offs = read_data(filename=filename)

    save_mrna_to_fasta(mRNAs, '../data/mrna_sequences.fasta')
    save_mrna_to_fasta(constant_part, '../data/constant_part.fasta')
    save_mrna_to_fasta(variable_part, '../data/variable_part.fasta')

    result = check_consistency(mRNAs, loop1s, switches, loop2s, stem1s, atgs, stem2s, linkers, post_linkers)

    # 输出结果
    for key, value in result.items():
        print(f"{key}: {value}")

    print('结果显示Switch、stem1和stem2是可变的，其余都是恒定的。')

    print(f'可变部分的长度为：stem1:{len(stem1s[0])}; stem2:{len(stem2s[0])}; Switvh:{len(switches[0])}')
    # features_array, labels_array = make_dataset_sequences_bio(mRNAs, ons, offs, on_offs)