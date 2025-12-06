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
    Convert an image to a base64-encoded string.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64-encoded string in the format 'data:image/png;base64,...'.
    """
    # Open image file
    img = Image.open(image_path)
    
    # Determine image format
    img_format = image_path.split('.')[-1].lower()
    if img_format == 'jpg':
        img_format = 'jpeg'  # PIL uses 'jpeg' instead of 'jpg'
    
    # Create a byte stream
    buffer = io.BytesIO()
    
    # Save image to byte stream
    img.save(buffer, format=img_format)
    
    # Get bytes and encode to base64
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    # Build data URI
    data_uri = f'data:image/{img_format};base64,{img_base64}'

    data_uri = {'img': data_uri}
    json_data = json.dumps(data_uri)
    
    return json_data


def plot_figure(df, figure_name, subsequence_length=2, top=10,
                spine_linewidth=2, width=4.5, alpha=0.7):
 
    values = df['fitness'].tolist()
    sequences = df['guide_rna'].apply(lambda x: x[20-subsequence_length:]).tolist()

    print(values)
    print(sequences)

    base_counts = Counter(sequences)

    # Find the most frequent subsequences
    most_common_bases = base_counts.most_common(top)

    # Print results
    names = []
    numbers = []

    print("Top 10 most frequent subsequences:")
    for base, count in most_common_bases:
        names.append(base)
        numbers.append(count)
        print(f"{base}: {count} times")
    

    # Collect values corresponding to each subsequence
    data = []
    for target_combination in names:
        matched_indices = [
            index for combination, index in zip(sequences, values)
            if combination == target_combination
        ]
        data.append(matched_indices)

    # Plot density curves
    plt.figure(figsize=(12, 8))
    for key, vals in zip(names, data):
        sns.kdeplot(
            data=vals,
            label=f'{key},n={len(vals)}',
            linewidth=width,
            alpha=1
        )

    # Set spine linewidth
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)
    
    # Thicken tick marks
    ax.tick_params(axis='both', width=spine_linewidth)

    # Labels
    plt.xlabel("Fitness", fontsize=28)
    plt.ylabel("Probability density", fontsize=28)

    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    plt.legend(fontsize=19)
    plt.legend(fontsize=19)
    plt.savefig(f'../result/{figure_name}.jpeg')
    plt.show()
    plt.close()

    return image_to
::contentReference[oaicite:0]{index=0}
