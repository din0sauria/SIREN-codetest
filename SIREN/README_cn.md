# SIREN（IEEE S&P 2025）

本仓库包含论文《Towards Reliable Verification of Unauthorized Data Usage in Personalized Text-to-Image Diffusion Models》（IEEE S&P 2025）的官方实现代码。

## 环境搭建指南
本节提供详细的环境准备和执行 SIREN 项目的步骤，请按照以下说明操作。

### 1. 环境搭建
#### 创建 Conda 环境：
使用 Python 3.9 创建一个名为 siren 的新 Conda 环境：
```bash
conda create --name siren python=3.9
```
#### 激活环境：
激活新创建的环境：
```bash
conda activate siren
```
### 2. 依赖安装
安装所需的 Python 包：
```bash
pip install -r requirements.txt
```

## 快速启动指南
### 1. 数据集与预训练模型准备
本项目支持本地数据集和 Hugging Face 数据集。
### 数据集
##### 本地数据集
可以使用 `--is_text` 标志指定是否读取文本。数据集中的文本-图像对应关系遵循命名约定，例如，`1.png` 对应于 `1.txt`。

##### Hugging Face 数据集
对于 Hugging Face 数据集，需要控制 `--load_text_key` 和 `--load_image_key` 参数。

### 模型
然后，从 [这里](https://www.dropbox.com/scl/fo/7cc8da2xiinfj6yrz670k/AIVqv4eQr5Xytosv-H4Iuq4?rlkey=4rl4tl3khjfr8310st29usw0e&st=cq0t5gwd&dl=0) 下载元学习编码器和检测器模型，并将它们放入 `./ckpt` 文件夹中。

### 2. 微调
#### 宝可梦编码器与解码器
为了方便起见，我们直接提供了在宝可梦数据集上微调后的编码器和解码器，可以从 [这里](https://www.dropbox.com/scl/fo/hr9c2y0a2kcujyqdgamm4/AIxk2wmF074xbf756guoONs?rlkey=d5vzavlvcitax005vlykty5e9&st=y4mngear&dl=0) 获取。

#### 微调扩散模型
在微调编码器和解码器之前，需要先训练一个个性化扩散模型。可以使用 [webui 代码](https://github.com/kohya-ss/sd-scripts) 来训练这个扩散模型。
我们将提供在宝可梦和 CelebA 数据集上训练的脚本。

##### 宝可梦
训练脚本：
```bash
accelerate launch --gpu_ids='GPU_ID'  train_network.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --dataset_config="pokemon.toml" \
    --output_dir="OUTPUT_PATH" \
    --output_name="OUTPUT_NAME" \
    --save_model_as=safetensors \
    --prior_loss_weight=1.0 \
    --max_train_epochs=80 \
    --learning_rate=1e-4 \
    --optimizer_type="AdamW" \
    --mixed_precision="fp16" \
    --save_every_n_epochs=20 \
    --network_module=networks.lora \
    --network_dim=64 \
    --gradient_checkpointing \
    --gradient_accumulation_steps=1 \
    --cache_latents
```
还需要一个数据集参数文件 `pokemon.toml`：
```toml
[general]
enable_bucket = true                        # 是否使用宽高比分桶

[[datasets]]
resolution = 512                            # 训练分辨率
batch_size = 4                              # 批量大小

  [[datasets.subsets]]
  image_dir = 'IMAGE_DATASET_PATH'                     # 指定包含训练图像的文件夹
  caption_extension = '.txt'            # 标注文件扩展名；如果使用 .txt 则更改此选项
  num_repeats = 1                          # 训练图像的重复次数
```

##### CelebA
训练脚本：
```bash
accelerate launch --gpu_ids='GPU_ID'  train_network.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --dataset_config="celeba.toml" \
    --output_dir="OUTPUT_PATH" \
    --output_name="OUTPUT_NAME" \
    --save_model_as=safetensors \
    --prior_loss_weight=1.0 \
    --max_train_epochs=80 \
    --learning_rate=1e-4 \
    --optimizer_type="AdamW" \
    --mixed_precision="fp16" \
    --save_every_n_epochs=20 \
    --network_module=networks.lora \
    --network_dim=64 \
    --gradient_checkpointing \
    --gradient_accumulation_steps=1 \
    --cache_latents
```
还需要一个数据集参数文件 `celeba.toml`：
```toml
[general]
enable_bucket = true                        # 是否使用宽高比分桶

[[datasets]]
resolution = 512                            # 训练分辨率
batch_size = 4                              # 批量大小

  [[datasets.subsets]]
  image_dir = 'IMAGE_DATASET_PATH'                     # 指定包含训练图像的文件夹
  caption_extension = '.txt'            # 标注文件扩展名；如果使用 .txt 则更改此选项
  num_repeats = 1                          # 训练图像的重复次数
```

#### 运行命令
以宝可梦为例，可以通过运行以下命令进行微调：
```bash
accelerate launch --gpu_ids='GPU_ID' fine_tune.py \
    --dataset_path "DATASET_FROM_LOCAL_OR_HUB" \
    --epoch 60 \
    --save_n_epoch 20 \
    --output_path "OUTPUT_PATH" \
    --log_dir "LOG_PATH" \
    --diffusion_path "runwayml/stable-diffusion-v1-5" \
    --lora_path "LORA_PATH" \
    --trigger_word "pokemon" \
    --decoder_checkpoint "META_DECODER_PATH" \
    --encoder_checkpoint "MATE_ENCODER_PATH" \
    --is_diffuser
```
请注意，这里的 `lora_path` 指的是从之前的微调扩散模型中获得的 LoRA。

### 3. 涂层（Coating）
#### 本地数据集
对于本地数据集，可以使用以下命令对图像应用涂层：
```bash
python coating.py \
    --dataset_path "DATASET_FROM_LOCAL" \
    --decoder_checkpoint "FINE_TUNE_DECODER_PATH" \
    --encoder_checkpoint "FINE_TUNE_ENCODER_PATH" \
    --output_path "OUTPUT_PATH" \
    --is_text \
    --gpu_id GPU_ID
```

#### Hugging Face 数据集
对于 Hugging Face 数据集，可以使用以下命令对图像应用涂层：
```bash
python coating.py \
    --dataset_path "DATASET_FROM_HUGGINGFACE" \
    --decoder_checkpoint "FINE_TUNE_DECODER_PATH" \
    --encoder_checkpoint "FINE_TUNE_ENCODER_PATH" \
    --output_path "OUTPUT_PATH" \
    --load_text_key "TEXT_KEY" \
    --load_image_key "IMAGE_KEY" \
    --gpu_id GPU_ID
```

### 4. 检测
#### 清洁数据集
可以使用原始清洁数据集训练一个个性化模型，然后使用该模型生成一个数据集作为清洁数据集。

对于检测清洁数据集，可以使用以下命令：
```bash
python detect.py \
    --dataset_path "CLEAN_DATASET" \
    --decoder_path "FINE_TUNE_DECODER_PATH" \
    --gpu_id GPU_ID \
    --output_path "OUTPUT_PATH" \
    --output_filename "CLEAN_DATASET_OUTPUT_FILENAME" 
```

#### 涂层数据集
可以使用涂层数据集训练一个个性化模型，然后使用该模型生成一个数据集作为可疑模型图像。

对于检测该数据集，可以使用以下命令：
```bash
python detect.py \
    --dataset_path "COATING_DATASET" \
    --decoder_path "FINE_TUNE_DECODER_PATH" \
    --gpu_id GPU_ID \
    --output_path "OUTPUT_PATH" \
    --output_filename "COATING_DATASET_OUTPUT_FILENAME" 
```

### 5. 假设检验
可以运行以下命令进行假设检验：
```bash
python ks_test.py \
    --clean_path "CLEAN_DATASET_OUTPUT_FILENAME" \
    --coating_path "COATING_DATASET_OUTPUT_FILENAME" \
    --output "OUTPUT_PATH" \
    --repeat 10000 \
    --samples 30
```

### 6. 元学习（可选）
#### 运行命令
可以直接使用元学习后的模型。如果想自行进行元学习，运行以下命令：
```bash
accelerate launch --gpu_ids='GPU_ID' meta.py \
   