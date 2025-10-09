import pandas as pd

def csv_to_fasta(csv_path, fasta_path, seq_column='sequence', id_prefix='seq'):
    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 检查是否存在指定列
    if seq_column not in df.columns:
        raise ValueError(f"列 '{seq_column}' 不存在于 CSV 文件中。")

    with open(fasta_path, 'w') as fasta_file:
        for i, seq in enumerate(df[seq_column]):
            # 写入FASTA格式
            fasta_file.write(f'>{id_prefix}_{i+1}\n')
            fasta_file.write(f'{seq.strip()}\n')

    print(f"已成功将 {len(df)} 条序列写入 {fasta_path}")

# 示例用法
csv_to_fasta('output.csv', 'output_sequences.fasta')
