# SIREN codetest

本仓库对论文《Towards Reliable Verification of Unauthorized Data Usage in Personalized Text-to-Image Diffusion Models》（IEEE S&P 2025）的代码仓库进行测试。

## 项目目录
```bash
CODETEST
├─annotated_subset # blip2标注后的数据集
├─blip2-opt-2___7b # 标注模型(gitignore)
├─coated_dataset # 经过SIREN处理的数据集和干净数据集
│  ├─coating #水印数据集
│  └─original #干净数据集
├─detection_results # 经过SIREN detect的结果
├─diffusions_lora_clean #干净数据集diffusions官方库lora微调权重
├─diffusions_lora_coated #水印数据集diffusions官方库lora微调权重
├─gen_pic #微调后模型生成的图片
│  ├─0
│  ├─1
│  ├─2
│  └─3
├─kohya_lora_clean #干净数据集kohya库lora微调权重
├─kohya_lora_coated #水印数据集kohya库lora微调权重
├─openai
│  └─clip-vit-large-patch14 #transfomer库需要的clip模型(gitignore)
├─report #测试报告
├─sd-scripts #kohya库
├─SIREN #SIREN库
├─stable-diffusion-v1-5 #stable-diffusion-v1-5官方模型(gitignore)
├─subset #测试子集
└─test_results #ks测试结果
caption.py #blip2标注脚本
clean.toml #干净数据集配置
coated.toml #水印数据集配置
gen_evaluate_pic.py # 生成图片和clip评价
gen_pics.py # 生成图片
README.md 
requirements.txt # 依赖包列表
train_dreambooth_lora.py #diffusion-v1-5官方lora实例
```

## 环境搭建指南
本节提供详细的环境准备和执行 SIREN 项目的步骤，请按照以下说明操作。

### 1. 环境搭建
#### 创建 Conda 环境：
使用 Python 3.9 创建一个名为 sirentest 的新 Conda 环境：
```bash
conda create --name sirentest python=3.9
```
#### 激活环境：
激活新创建的环境：
```bash
conda activate sirentest
```
### 2. 依赖安装
安装所需的 Python 包：
```bash
pip install -r requirements.txt
```
支持GPU的torch版本可能需要自行选择，本实验使用的torch版本为torch2.4.1+cu118

注意执行diffusions官方库训练时需要使用最新diffusers库而不是diffusers发布版
```bash
pip install -U git+https://github.com/huggingface/diffusers
```
### 3. 模型下载

由于模型文件太大未上传到仓库，可以在huggingface或镜像站下载。
必要的模型包括`blip2-opt-2.7b`,`clip-vit-large-patch14`,`stable-diffusion-v1-5`。
注意`clip-vit-large-patch14`是transformer库需要的clip模型，需要根据库函数配置。注意文件夹命名，如`blip2-opt-2.7`因为不能有`.`需命名为`blip2-opt-2___7b`

## 快速复现指南

### 数据预处理

#### 下载 Anime-Chibi 数据集并随机筛选 500 张图片

在下载目录下执行以下代码，然后将子集subset移动到目录下
```python
import os
import shutil
import random

# 原始数据集路径
source_dir = "./download"
# 测试子集路径
target_dir = "./subset"

os.makedirs(target_dir, exist_ok=True)

image_files = [f for f in os.listdir(source_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

selected_images = random.sample(image_files, 500)

for img in selected_images:
    shutil.copy(os.path.join(source_dir, img), target_dir)
```

#### 使用 BLIP-2 生成 Caption

下载blip2-opt-2.7b到目录，运行`caption.py`进行推理生成 Caption


#### 添加 SIREN 水印

```bash
python ./SIREN/coating.py --dataset_path "./annotated_subset" --decoder_checkpoint "./SIREN/ckpt/pokemon_decoder.pth" --encoder_checkpoint "./SIREN/ckpt/pokemon_encoder.pth" --output_path "./coated_dataset" --is_text --gpu_id 0
```


### 训练个性化模型与生成图片

#### 使用 kohya-ss LoRA 仓库训练

    训练使用NVIDIA GeForce RTX 4060 Laptop GPU，显存8G专用+8G共享=16G
    由于本实验以学习为目的，基于显存及速度考虑，调整了微调参数使显存更低速度更快

下载kohya-ss LoRA 仓库保存到sd-scripts

运行训练命令（windous PS使用`换行，下同）
```bash
accelerate launch --gpu_ids='0'  sd-scripts/train_network.py `
    --pretrained_model_name_or_path="stable-diffusion-v1-5" `
    --dataset_config="clean.toml" `
    --output_dir="./kohya_lora_clean" `
    --output_name="kohya_lora_clean" `
    --save_model_as=safetensors `
    --prior_loss_weight=1.0 `
    --max_train_epochs=5 `
    --learning_rate=1e-4 `
    --optimizer_type="AdamW8bit" `
    --mixed_precision="no" `
    --save_every_n_epochs=10 `
    --network_module=networks.lora `
    --network_dim=16 `
    --network_alpha=8 `
    --gradient_accumulation_steps=1 `
    --cache_latents 
    
accelerate launch --gpu_ids='0'  sd-scripts/train_network.py `
    --pretrained_model_name_or_path="stable-diffusion-v1-5" `
    --dataset_config="coated.toml" `
    --output_dir="./kohya_lora_coated" `
    --output_name="kohya_lora_coated" `
    --save_model_as=safetensors `
    --prior_loss_weight=1.0 `
    --max_train_epochs=5 `
    --learning_rate=1e-4 `
    --optimizer_type="AdamW8bit" `
    --mixed_precision="fp16" `
    --save_every_n_epochs=10 `
    --network_module=networks.lora `
    --network_dim=16 `
    --network_alpha=8 `
    --gradient_accumulation_steps=1 `
    --cache_latents 
```



#### 使用 diffusers 官方仓库训练

```bash
accelerate launch train_dreambooth_lora.py `
  --pretrained_model_name_or_path="stable-diffusion-v1-5" `
  --output_dir="./diffusions_lora_clean" `
  --instance_data_dir="coated_dataset/original"
  --instance_prompt="a picture of a cute anime character" `
  --resolution=512 `
  --train_batch_size=1 `
  --gradient_accumulation_steps=1 `
  --learning_rate=1e-4 `
  --num_train_epochs=5 `
  --validation_prompt="a picture of a cute anime character" `
  --validation_epochs=10 `
  --rank=16 `
  --seed=42 `
  --use_8bit_adam

accelerate launch train_dreambooth_lora.py `
  --pretrained_model_name_or_path="stable-diffusion-v1-5" `
  --output_dir="./diffusions_lora_coated" `
  --instance_data_dir="coated_dataset/coating"
  --instance_prompt="a picture of a cute anime character" `
  --resolution=512 `
  --train_batch_size=1 `
  --gradient_accumulation_steps=1 `
  --learning_rate=1e-4 `
  --num_train_epochs=5 `
  --validation_prompt="a picture of a cute anime character" `
  --validation_epochs=10 `
  --rank=16 `
  --seed=42 `
  --use_8bit_adam
```

#### 生成图片

运行`gen_evaluate_pic.py`用四种权重分别生成图片，自动命名为{modelIndex}_{promptIndex}.png
由于本实验已经引入clip，可以方便地使用clip计算图文相似度分数


### 检测图片

#### 检测水印

再运行`gen_pics.py`每个模型生成多个图像，保存在./gen_pic/{modelIndex}文件夹下，并使用SIREN模型检测水印。

```bash
# 检测干净数据集
python ./SIREN/detect.py `
    --dataset_path "./gen_pic/0" `
    --decoder_path "./SIREN/ckpt/pokemon_decoder.pth" `
    --gpu_id 0 `
    --output_path "./detection_results" `
    --output_filename "0.npy"

python ./SIREN/detect.py `
    --dataset_path "./gen_pic/2" `
    --decoder_path "./SIREN/ckpt/pokemon_decoder.pth" `
    --gpu_id 0 `
    --output_path "./detection_results" `
    --output_filename "2.npy"

# 检测加水印数据集
python ./SIREN/detect.py `
    --dataset_path "./gen_pic/1" `
    --decoder_path "./SIREN/ckpt/pokemon_decoder.pth" `
    --gpu_id 0 `
    --output_path "./detection_results" `
    --output_filename "1.npy"

python ./SIREN/detect.py `
    --dataset_path "./gen_pic/3" `
    --decoder_path "./SIREN/ckpt/pokemon_decoder.pth" `
    --gpu_id 0 `
    --output_path "./detection_results" `
    --output_filename "3.npy"
```

#### 假设检验

```bash
python ./SIREN/ks_test.py `
    --clean_path "./detection_results/0.npy" `
    --coating_path "./detection_results/1.npy" `
    --output "./test_results/kohya.log"
    --repeat=100000
    --samples=50

python ./SIREN/ks_test.py `
    --clean_path "./detection_results/2.npy" `
    --coating_path "./detection_results/3.npy" `
    --output "./test_results/diffusers.log" 
    --repeat=100000
    --samples=50
```
