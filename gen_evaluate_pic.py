from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import requests

# 加载 Stable Diffusion 模型
pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# 加载 CLIP 模型和处理器
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# 定义权重文件和提示文本
weights = [
    "./kohya_lora_clean/kohya_lora_clean.safetensors",
    "./kohya_lora_coated/kohya_lora_coated.safetensors",
    "./diffusions_lora_clean/pytorch_lora_weights.safetensors",
    "./diffusions_lora_coated/pytorch_lora_weights.safetensors"
]
prompts = [
    "a girl hugging a cat with long hair",
    "a cute anime girl holding a teddy bear and wearing a pink dress",
    "a cute cartoon girl with long hair and a white dress"
]

# 确保生成图片的目录存在
os.makedirs("gen_pic", exist_ok=True)

# 生成图片并计算图文相似度
for i in range(len(weights)):
    pipe.load_lora_weights(weights[i])
    for j in range(len(prompts)):
        # 生成图片
        image = pipe(prompts[j]).images[0]
        image_path = f"gen_pic/{i}_{j}.png"
        image.save(image_path)
        
        # 计算图文相似度
        try:
            # 加载本地图片
            image = Image.open(image_path)
            
            # 处理输入
            inputs = clip_processor(
                text=[prompts[j]],  # 使用对应的提示文本
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            # 计算相似度
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image  # 图文相似度分数
            probs = logits_per_image.softmax(dim=1)      # 转换为概率
            
            print(f"Image {i}_{j}.png - Text: {prompts[j]}")
            print(f"Similarity Score: {logits_per_image.item():.4f}")
            print(f"Probability: {probs.item():.4f}\n")
            
        except Exception as e:
            print(f"Error processing image {i}_{j}.png: {e}")