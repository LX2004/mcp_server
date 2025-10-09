#!/usr/bin/env python
# Author: Yicheng yang [achensparrow@gmail.com]
# Datatime: 2021-01-26 16:00
# Filename: final_compute_gammasAfitness.py
# Description:
# Used to calculate fitness and genetic necessity of genes in Crispri-seq
import argparse
import logging
import pathlib
import sys
from typing import Tuple, Set

import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

_PACKAGEDIR = pathlib.Path(__file__).parent

# 常量配置
class Config:
    NORM_SIZE = 40_000_000.0 
    MIN_START_READS = 100
    PSEUDO = 1
    REQUIRED_COLUMNS = {'guide_rna', 'start_count', 'end_count'}

def parse_args() -> argparse.Namespace:
    """解析命令行参数并验证参数组合"""
    parser = argparse.ArgumentParser(
        description='Calculate fitness and genetic necessity of genes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        '--guide_rnafile', type=pathlib.Path,
        help='File containing single guide RNA and six counts number (TSV format)',
        default=_PACKAGEDIR / '../data/fitness_input_sample.tsv')
    parser.add_argument(
        '--fitnessfile', type=pathlib.Path,
        help='Output file for fitness measurements',
        default=None)
    parser.add_argument(
        '--growth', type=int,
        help='Number of generations grown (g*t value)',
        default=10)
    
    args = parser.parse_args()
    
    if not args.guide_rnafile.exists():
        raise FileNotFoundError(f"Target file not found: {args.guide_rnafile}")
    
    if args.growth <= 0:
        raise ValueError("Growth generations must be positive integer")
    
    if args.fitnessfile is None:
        args.fitnessfile = _PACKAGEDIR / '../data/fitness111.tsv'
    else:
        args.fitnessfile.parent.mkdir(parents=True, exist_ok=True)
    
    return args

def validate_dataframe(df: pd.DataFrame, required_cols: Set[str]) -> Tuple[bool, str]:
    """验证DataFrame的列完整性"""
    missing = required_cols - set(df.columns)
    if missing:
        return False, f"Missing required columns: {', '.join(sorted(missing))}"
    return True, ""

def process_count_columns(df: pd.DataFrame) -> pd.DataFrame:
    """处理count列转换为数值列表并计算均值"""
    df = df.copy()
    df['start_mean'] = df['start_count'].apply(
        lambda x: np.mean(list(map(int, x.split(',')))))
    df['end_mean'] = df['end_count'].apply(
        lambda x: np.mean(list(map(int, x.split(',')))))
    return df

def compute_pvalues(df: pd.DataFrame) -> pd.DataFrame:
    """执行配对t检验计算p值"""
    p_values = []
    for _, row in df.iterrows():
        start = list(map(int, row['start_count'].split(',')))
        end = list(map(int, row['end_count'].split(',')))
        if len(start) == len(end) and len(start) > 1:
            _, p_val = ttest_rel(end, start)
        else:
            p_val = float('nan')
        p_values.append(p_val)
    df['p_value'] = p_values
    return df

def adjust_pvalues(df: pd.DataFrame) -> pd.DataFrame:
    """执行Benjamini-Hochberg校正"""
    df = df.copy()
    reject, padj, _, _ = multipletests(
        df['p_value'].fillna(1), 
        method='fdr_bh'
    )
    df['padj'] = padj
    return df

def load_guide_rnas(guide_rnafile: pathlib.Path) -> pd.DataFrame:
    """加载并预处理靶标文件"""
    logger.info(f"Loading guide_rna file: {guide_rnafile}")
    try:
        df = pd.read_csv(
            guide_rnafile,
            sep='\t',
            dtype={'guide_rna': 'category'}
        )
    except pd.errors.EmptyDataError:
        raise ValueError("Target file is empty or corrupt")
    
    is_valid, msg = validate_dataframe(df, Config.REQUIRED_COLUMNS)
    if not is_valid:
        raise ValueError(msg)
    
    df = process_count_columns(df)
    return df


def load_tsv(tsv) -> pd.DataFrame:
    

    try:
        tsv
        tsv['guide_rna'] = tsv['guide_rna'].astype('category')
        
    except pd.errors.EmptyDataError:
        raise ValueError("Target file is empty or corrupt")
    
    is_valid, msg = validate_dataframe(tsv, Config.REQUIRED_COLUMNS)
    if not is_valid:
        raise ValueError(msg)
    
    tsv = process_count_columns(tsv)
    return tsv

def compute_normalized_log_counts(df: pd.DataFrame, count_col: str) -> pd.Series:
    """标准化处理并返回对数计数值"""
    logger.debug(f"Processing counts from {count_col}")
    # df = df.set_index('guide_rna')
    total_reads = df[count_col].sum()
    if total_reads == 0:
        raise ValueError("Total reads sum to zero")
    
    norm_factor = Config.NORM_SIZE / total_reads
    normalized = df[count_col] * norm_factor
    return np.log2(normalized.clip(Config.PSEUDO))

def compute_gamma(
    df: pd.DataFrame,
    controlset: Set[str],
    gt: int
) -> pd.DataFrame:
    """核心计算逻辑"""
    logger.info("Computing gamma values")
    try:
        # 计算标准化对数计数
        df = df.set_index('guide_rna')
        start = compute_normalized_log_counts(df, 'start_mean')
        end = compute_normalized_log_counts(df, 'end_mean')

        # 生成起始终止掩码
        start_mask = df['start_mean'] > Config.MIN_START_READS
        start_mask.name = 'start_mask'
        end_mask = df['end_mean'] > Config.MIN_START_READS
        end_mask.name = 'end_mask'
        
        # 计算差异与去中心化
        diff = end - start
        diff = diff.where(start_mask, np.nan)
        diff = diff.where(end_mask, np.nan)
        
        # 计算gamma值
        center = diff.loc[diff.index.isin(controlset)].median()
        gamma = (diff - center) / gt
        gamma.name = 'gamma'
        
        # 合并结果
        result = pd.DataFrame({
            'gamma': gamma,
            'start_mask': start_mask
        })
        result.index.name = 'guide_rna'
        return result
    except Exception as e:
        logger.error("Gamma computation failed", exc_info=True)
        raise

def assign_random_coordinates(gene_data):
    import random
    # 创建新列表避免修改原始数据
    result = []
    for item in gene_data:
        # 复制原始字典并添加新坐标
        new_item = item.copy()
        new_item['xField'] = random.uniform(0, 10)  # 生成 0-10 的随机浮点数
        new_item['yField'] = random.uniform(0, 10)
        result.append(new_item)
    return result


def add_necessity_column(df):
    """添加必要性判断列"""
    df = df.copy()
    
    # 处理无穷值和零值
    with np.errstate(divide='ignore', invalid='ignore'):
        df['log2FoldChange'] = np.log2(df['FoldChange'])
    
    # 创建判断条件（自动处理NaN）
    is_essential = (df['log2FoldChange'] < -1) & (df['padj'] < 0.05)
    # df['Necessity'] = np.where(is_essential, 'essential', 'neutral')
    df['essential'] = is_essential
    
    # 调整列顺序
    # column_order = ['locus_tag', 'gene', 'guide_rna', 'Necessity', 'FoldChange', 'log2FoldChange', 'p_value', 'padj']
    # return df[column_order]
    return df

from io import StringIO

# 1. 读取 TSV 文件并将其转化为字符串
def read_tsv_to_string(file_path):
    # 读取 TSV 文件
    df = pd.read_csv(file_path, sep='\t')
    
    # 将 DataFrame 转换为字符串，index=False 不包含行索引
    df_string = df.to_string(index=False)
    
    return df_string

# 2. 将字符串转化为 DataFrame 并保存为 TSV 文件
def string_to_df_and_save_tsv(df_string, output_file_path):
    # 使用 StringIO 将字符串转化为类似文件的对象
    data = StringIO(df_string)
    
    # 将字符串读取为 DataFrame
    df = pd.read_csv(data, sep='\t')
    
    # 保存为 TSV 文件
    df.to_csv(output_file_path, sep='\t', index=False)

    print(f"TSV 文件已保存为 {output_file_path}")


def compute_main(tsv=None, growth=None):
    try:
        print('start calculate fitness!!!')

        args = parse_args()
        logger.info("Pipeline started")
        # print('start calculate fitness!!!')

        if tsv:

            print('tsv = \n', tsv)
            df = StringIO(tsv)

            guide_rnas_df = pd.read_csv(df, sep='\t')

            print('guide_rnas_df = \n', guide_rnas_df)
            guide_rnas_df = load_tsv(tsv=guide_rnas_df)
        
        else: 
            guide_rnas_df = load_guide_rnas(args.guide_rnafile)
        
         
        controls = set(guide_rnas_df['guide_rna'].unique())
        
        # 计算gamma和fitness

        if growth:
            gamma_df = compute_gamma(guide_rnas_df, controls, growth)

        else:
            gamma_df = compute_gamma(guide_rnas_df, controls, args.growth)

        gamma_df = gamma_df.reset_index()
        
        # 计算统计量
        processed_df = compute_pvalues(guide_rnas_df)
        processed_df = adjust_pvalues(processed_df)
        
        # 合并所有结果
        final_df = pd.merge(
            guide_rnas_df,
            gamma_df,
            on='guide_rna'
        )
        final_df = pd.merge(
            final_df,
            processed_df[['guide_rna', 'p_value', 'padj']],
            on='guide_rna'
        )
        final_df['fitness'] = final_df['gamma'] + 1

        # 生成FoldChange
        final_df['FoldChange'] = final_df['end_mean'] / final_df['start_mean']

        # 添加必要性判断列
        final_df = add_necessity_column(final_df)
        df = final_df.drop(columns=['start_count', 'end_count'])
        print('df = ', df)

        columns_to_convert = ['start_mean', 'end_mean', 'p_value_x', 'gamma', 'p_value_y', 'padj', 'fitness', 'FoldChange', 'log2FoldChange']
        # 确保所有指定列转换为 float 类型
        df[columns_to_convert] = df[columns_to_convert].astype(float)

        gene_essential = {}

        for gene, ess in zip(df["gene"], df["essential"]):

            if gene in gene_essential:
                gene_essential[gene] = ess and gene_essential[gene]

            else:
                gene_essential[gene] = ess

        gene_fitness = {}

        for gene, fit in zip(df["gene"], df["fitness"]):

            if gene in gene_fitness:
                gene_fitness[gene] = (fit + gene_fitness[gene]) / 2

            else:
                gene_fitness[gene] = fit

        # csv数据转的字符串
        gene_ess_fit_Graph = [{"gene": label, "essential": gene_essential[label], "fitness": round(gene_fitness[label], 4), "sizeField": round(abs(gene_fitness[label])*20, 4), "seriesField": 0 if gene_fitness[label] > 0 else 1} for label in gene_essential]
        # gene_ess_fit_Graph = [{"gene": label, "essential": "TRUE" if gene_essential[label] else "FALSE", "fitness": round(gene_fitness[label], 4), "sizeField": round(abs(gene_fitness[label])*20, 4), "seriesField": 0 if gene_fitness[label] > 0 else 1} for label in gene_essential]
        gene_ess_fit_Graph = assign_random_coordinates(gene_data=gene_ess_fit_Graph)

        # df = df.rename(columns={'fitness': 'calculated_fitness'})
        csv = df.to_csv(index=False)
        csv = csv.replace('True', 'TRUE').replace('False', 'FALSE')

        return gene_ess_fit_Graph, csv

    except Exception as e:
        logger.critical(f"Pipeline failed: {str(e)}", exc_info=True)
        return 1
    


def plot_gene_scatter(gene_data):
    
    import matplotlib.pyplot as plt
    import numpy as np

    # 准备绘图数据
    x = [item['xField'] for item in gene_data]
    y = [item['yField'] for item in gene_data]
    areas = [abs(item['sizeField']) * 200 for item in gene_data]  # 面积缩放系数
    colors = ['red' if item['sizeField'] > 0 else 'blue' for item in gene_data]
    
    # 创建画布
    plt.figure(figsize=(10, 8))
    
    # 绘制散点
    scatter = plt.scatter(
        x, y,
        s=areas,
        c=colors,
        alpha=0.6,
        edgecolors='w',
        linewidths=0.5
    )
    
    # 添加图例
    red_patch = plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor='red', markersize=8, label='Fitness > 0')
    blue_patch = plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor='blue', markersize=8, label='Fitness < 0')
    plt.legend(handles=[red_patch, blue_patch])
    
    # 设置坐标轴
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.title('Gene Fitness Scatter Plot', fontsize=14)
    
    # 显示网格
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 显示图表
    plt.savefig('../result/gene_essential.png')
    plt.show()

# 使用示例（接续之前的坐标分配函数）


if __name__ == "__main__":

#     data_string = """locus_tag\tgene\tguide_rna\tstart_count\tend_count
# BSU00010\tdnaA\tCCCCCCCCCCCCCCCCCCC\t218,223,211\t22542,23242,22246
# BSU00020\tcnaB\tTTTTTTTTTTTTTTTTTTTT\t101,101,121\t542,5562,512
# BSU00030\tcnaC\tAAAAAAAAAAAAAAAAAAAA\t523,532,552\t92899,92899,92900
# BSU00040\tcnaD\tCCCCCCCCCCCCCCCCCTT\t91002,90200,10201\t88103,89109,89101
# BSU00050\tqnaA\tCCCCCCCCCACCCCCCCCAA\t22542,23242,22246\t218,223,201
# BSU00060\tqnaC\tCCCCCCCCCACCCCCCCCTT\t22553,23242,22246\t200,203,121
# BSU00070\tqnaF\tCCCCCCCCCACCCCCCCCAT\t22653,23242,28246\t200,203,101
# BSU00080\tcnaB\tTTTTTTTTTTTTTTTTTCAT\t22653,23242,28246\t200,203,111"""

    data_string ="""locus_tag\tgene\tguide_rna\tstart_count\tend_count
BSU00010\tdnaA\tTCCCCCCCCCCCCCCCCCCC\t218,223,211\t22542,23242,22246
BSU00020\tcnaB\tTTTTTTTTTTTTTTTTTTTT\t101,101,121\t542,5562,512
BSU00030\tcnaC\tAAAAAAAAAAAAAAAAAAAA\t523,532,552\t92899,92899,92900
BSU00040\tcnaD\tCCCCCCCCCCCCCCCCCCAT\t91002,90200,10201\t1399,1299,1401
BSU00050\tcnaD\tACCCCCCCCCCCCCCCCCTT\t91002,90200,10201\t139,129,101"""

    # print(sys.exit(compute_main(tsv=data_string)))
    gene_ess_fit_Graph, csv = compute_main(tsv=data_string, growth=20)
    plot_gene_scatter(gene_data = gene_ess_fit_Graph)

    print('csv = \n', csv)
    print(' gene_ess_fit_Graph = ', gene_ess_fit_Graph)
