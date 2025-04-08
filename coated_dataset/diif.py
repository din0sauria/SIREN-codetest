import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def generate_difference_map(img1_path, img2_path, output_path):

    # 加载图片
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')

    print(img1.size)

    # 确保图片尺寸一致
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)

    # 转换为numpy数组
    img1_array = np.array(img1)
    img2_array = np.array(img2)

    # 计算差距图
    diff_array = np.abs(img1_array - img2_array)

    # 将差距图转换为图像
    diff_image = Image.fromarray(diff_array.astype(np.uint8))

    # 保存差距图
    #diff_image.save(output_path)
    #print(f"Difference map saved to {output_path}")

    # 可视化
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(img1)
    plt.title("Image 1")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img2)
    plt.title("Image 2")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(diff_image)
    plt.title("Difference Map")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

img1_path = r"coated_dataset\coating\1.png"
img2_path = r"coated_dataset\original\1.png"  
output_path = "difference_map.png"

generate_difference_map(img1_path, img2_path, output_path)