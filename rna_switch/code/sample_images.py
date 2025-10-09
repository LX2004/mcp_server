import argparse
import torch
import torchvision
from utils import *
import script_utils
import os

def get_sequence_list_from_csv(file_path):
    # 读取指定列
    df = pd.read_csv(file_path, usecols=["switch", "stem1", "stem2"])
    
    # 将每行三个元素拼接成一个字符串，形成列表
    sequence_list = df.apply(lambda row: row['switch'] + row['stem1'] + row['stem2'], axis=1).tolist()
    
    return sequence_list


def main():

    args = create_argparser().parse_args()

    '''准备数据集'''
    nat = get_sequence_list_from_csv(file_path='/home/leixin/mcp_server/switch/data/Toehold_mRNA_Dataset_cleanplus.csv')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(4)

    try:
        for epoch in range(700,1200,100):

            diffusion = script_utils.get_diffusion_from_args(args).to(device)
            model_path = f'/home/liangce/lx/Promoter_mRNA_synthetic/Synthesizing_mRNA/result/Switch-ddpm-2025-07-10-21-08-iteration-{epoch}--kernel=3--model.pth'
            
            print(' model_path', model_path)

            diffusion.load_state_dict(torch.load(model_path))
            sequences = []

            for i in range(2):

                print('strat to generate sequences')
                samples = diffusion.sample(args.num_images, device)
                print('end to generate sequences')

                print('samples.shape = ', samples.shape)
                samples = samples.squeeze(dim=1)
                print('samples.shape = ', samples.shape)

                samples = samples.to('cpu').detach().numpy()

                for i in range(samples.shape[0]):

                    decoded_sequence = decode_one_hot(samples[i])
                    sequences.append("A" + decoded_sequence)

            k_mer_fre_cor = calculate_overall_kmer_correlation(dataset1=sequences, dataset2=nat, k=6)

            print('6_mer_fre_cor = ', k_mer_fre_cor)
                    # os.remove(model_path)
                    # continue

                # make_fasta_file(sequences,path=f'../sequences/Not_all_promoter-ddpm-2024-04-22-15-58-iteration-{epoch}_6_mer_fre_cor={k_mer_fre_cor}.fasta') # 将序列写入fasta文件。
            make_fasta_file(sequences,path=f'../sequence/Switche_epoch={epoch}_6_mer_fre_cor={k_mer_fre_cor}.fasta') # 将序列写入fasta文件。

    except KeyboardInterrupt:
        print("Keyboard interrupt, generation finished early")


def create_argparser():

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    defaults = dict(num_images=1024, device=device, schedule_low=1e-4,
    schedule_high=0.02,out_init_conv_padding = 1)
    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--save_dir", type=str)
    script_utils.add_dict_to_argparser(parser, defaults)

    return parser

if __name__ == "__main__":
    main()