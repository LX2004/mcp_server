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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

_PACKAGEDIR = pathlib.Path(__file__).parent

# Constants config
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
        lambda x: np.mean(list(map(int, x.sp
