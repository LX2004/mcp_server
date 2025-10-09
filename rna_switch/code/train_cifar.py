import argparse
import datetime
import torch
import numpy as np
from torch.utils.data import DataLoader,Dataset
# from torchvision import datasets
import script_utils
from torch.optim.lr_scheduler import StepLR
from utils import *
import pdb

'''继承数据集，自定义所需的数据集'''
class CustomDataset(Dataset):
    def __init__(self, data_folder):
        self.data = data_folder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.tensor(sample, dtype=torch.float32)


def get_sequence_list_from_csv(file_path):
    # 读取指定列
    df = pd.read_csv(file_path, usecols=["switch", "stem1", "stem2"])
    
    # 将每行三个元素拼接成一个字符串，形成列表
    sequence_list = df.apply(lambda row: row['switch'] + row['stem1'] + row['stem2'], axis=1).tolist()
    
    return sequence_list

def main():

    loss_flag = 0.15 # loss低于该值开始存储模型，只存储一个损失最低的模型
    test_loss_save = []

    args = create_argparser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(7)

    try:
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)

        if args.model_checkpoint is not None:
            diffusion.load_state_dict(torch.load(args.model_checkpoint))
        if args.optim_checkpoint is not None:
            optimizer.load_state_dict(torch.load(args.optim_checkpoint))

        batch_size = args.batch_size

        # 定义学习率调度器，每100回合将学习率乘以0.5
        scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

        '''准备数据集'''
        all_sample = get_sequence_list_from_csv(file_path='/home/liangce/lx/Promoter_mRNA_synthetic/data/Toehold_mRNA_Dataset_cleanplus.csv')
        train_size = int(0.8 *  len(all_sample))  # 90% 用于训练集
        print(all_sample[0],len(all_sample[0]))
        
        # 对训练集进行one-hot编码
        encoded_sequence_train = []

        for sequence in all_sample[:train_size]:

            if len(sequence) != 45:
                print('error!!!')

            encoded_sequence = one_hot_encoding(sequence[1:]) # 舍弃了第一个A
            encoded_sequence_train.append(encoded_sequence)
            
        # 对测试集进行one-hot编码
        encoded_sequence_test = []
        for sequence in all_sample[train_size:]:

            if len(sequence) != 45:
                print('error!!!')
            
            encoded_sequence = one_hot_encoding(sequence[1:])
            encoded_sequence_test.append(encoded_sequence)

        train_arrary = np.array(encoded_sequence_train)
        test_arrary = np.array(encoded_sequence_test)

        print('train_arrary.shape = ', train_arrary.shape)
        print('test_arrary.shape = ', test_arrary.shape)

        train_arrary = np.expand_dims(train_arrary, axis=1)
        test_arrary = np.expand_dims(test_arrary, axis=1)

        # 检查形状
        print("插入维度后的 train_array 形状：", train_arrary.shape)
        print("插入维度后的 test_array 形状：", test_arrary.shape)
        
        train_dataset = CustomDataset(train_arrary)
        test_dataset = CustomDataset(test_arrary)

        '''#创建数据加载器 '''

        # 创建训练集和测试集的DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        acc_train_loss = 0

        for iteration in range(1, args.iterations + 1):

            # train_loss_epoch = 0

            diffusion.train()

            # 开始训练
            for x in train_loader:
                # x, y = next(train_loader)
                x = x.to(device)
                # y = y.to(device)
                
                loss = diffusion(x)

                acc_train_loss += loss.item()
                # train_loss_epoch += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                diffusion.update_ema()

            print(f'epoch ={iteration}, train loss = {acc_train_loss}')

            # 更新学习率
            scheduler.step()
            
            if iteration % args.log_rate == 0:
                test_loss = 0

                with torch.no_grad():

                    diffusion.eval()
                    for x in test_loader:

                        x = x.to(device)
                        loss = diffusion(x)
                        test_loss += loss.item()
                
                if args.use_labels:
                    samples = diffusion.sample(10, device, y=torch.arange(10, device=device))
                else:
                    samples = diffusion.sample(10, device)
                
                samples = ((samples + 1) / 2).clip(0, 1).permute(0, 2, 3, 1).numpy()

                test_loss /= len(test_loader)
                acc_train_loss /= args.log_rate
            
                print(f'epoch = {iteration}, test_loss = {test_loss}')
            
            acc_train_loss = 0
            test_loss_save.append(test_loss)

            if test_loss < loss_flag:

                loss_flag = test_loss
                print('save best model')

                # model_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-kernel={1+2*args.out_init_conv_padding}--best-model.pth"
                # torch.save(diffusion.state_dict(), model_filename)

            if iteration % args.checkpoint_rate == 0:

                test_loss_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}--kernel={1+2*args.out_init_conv_padding}--test_loss.npy"
                model_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}--kernel={1+2*args.out_init_conv_padding}--model.pth"
                np.save(test_loss_filename, np.array(test_loss_save))

                
                optim_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-optim.pth"

                torch.save(diffusion.state_dict(), model_filename)
                torch.save(optimizer.state_dict(), optim_filename)

                # 保存损失函数变化趋势


    except KeyboardInterrupt:

        print("Keyboard interrupt, run finished early")


def create_argparser():
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    run_name = datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M")

    defaults = dict(

        learning_rate=1e-4,
        batch_size=512,
        iterations=2000,

        log_to_wandb=True,
        log_rate=1,
        # checkpoint_rate=200,
        checkpoint_rate=100,
        log_dir="../result",
        project_name='Switch',

        out_init_conv_padding = 1,
        run_name=run_name,

        model_checkpoint=None,
        optim_checkpoint=None,

        schedule_low=1e-4,
        schedule_high=0.02,

        device=device,
    )

    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()