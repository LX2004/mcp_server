from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
import pandas as pd
import json
import base64
from PIL import Image
import io

def image_to_base64(image_path):
    """
    将图片转换为base64字符串
    
    参数:
        image_path (str): 图片文件的路径
        
    返回:
        str: 格式为 'data:image/png;base64,...' 的base64字符串
    """
    # 打开图片文件
    img = Image.open(image_path)
    
    # 确定图片格式
    img_format = image_path.split('.')[-1].lower()
    if img_format == 'jpg':
        img_format = 'jpeg'  # PIL使用'jpeg'而不是'jpg'
    
    # 创建一个字节流
    buffer = io.BytesIO()
    
    # 保存图片到字节流
    img.save(buffer, format=img_format)
    
    # 获取字节流的内容并进行base64编码
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    # 构建完整的data URI
    data_uri = f'data:image/{img_format};base64,{img_base64}'

    data_uri = {'img':data_uri}
    json_data = json.dumps(data_uri)
    
    # return data_uri
    return json_data



def plot_figure(df, figure_name, subsequence_length=2, top = 10,spine_linewidth=2,width=4.5,alpha=0.7):
 
    values = df['fitness'].tolist()
    sequences = df['guide_rna'].apply(lambda x: x[20-subsequence_length:]).tolist()

    print(values)
    print(sequences)

    base_counts = Counter(sequences)

    # 找到出现频率最高的五个碱基组合
    most_common_bases = base_counts.most_common(top)

    # 输出结果
    names = []
    numbers = []

    print("出现频率最高的10个碱基组合:")
    for base, count in most_common_bases:
        names.append(base)
        numbers.append(count)
        print(f"{base}: {count}次")
    

    # 要查找的碱基组合
    data = []
    for target_combination in names:

        # 使用列表推导式和zip函数同时遍历两个列表，将符合条件的指标存储在一个新的列表中
        matched_indices = [index for combination, index in zip(sequences, values) if combination == target_combination]
        data.append(matched_indices)

    # 绘制分布曲线
    plt.figure(figsize=(12, 8))
    for key, values in zip(names, data):
        sns.kdeplot(data=values, label=f'{key},n={len(values)}', linewidth=width, alpha=1)  # 使用 seaborn 库的 kdeplot 方法绘制概率密度函数图



    # 设置边框粗细
    ax = plt.gca()  # 获取当前的轴对象
    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)
    
    # 设置刻度线加粗
    ax.tick_params(axis='both', width=spine_linewidth)  # 刻度线宽度

    # 设置图形标题和标签
    # plt.title("Probability Density Function of guide rna")
    plt.xlabel("Fitness", fontsize=28)
    plt.ylabel("Probability density", fontsize=28)

    plt.xticks(fontsize=24)  # 设置 x 轴刻度字体大小
    plt.yticks(fontsize=24)  # 设置 y 轴刻度字体大小

    plt.legend(fontsize=19)
    plt.legend(fontsize=19)
    plt.savefig(f'../result/{figure_name}.jpeg')
    # plt.savefig(f'paper_figure/{figure_name}.png')
    plt.show()
    plt.close()

    return image_to_base64(image_path=f'../result/{figure_name}.jpeg')
