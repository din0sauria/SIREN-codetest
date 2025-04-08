import torch
from PIL import Image
import os
from transformers import Blip2Processor, Blip2ForConditionalGeneration

model_path = "./blip2-opt-2___7b"
dataset_path = "./subset"
output_dataset_path = "./annotated_subset"

processor = Blip2Processor.from_pretrained(model_path)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_path, 
    device_map="auto",
    torch_dtype=torch.float16
)

# 输出目录
os.makedirs(output_dataset_path, exist_ok=True)

for filename in os.listdir(dataset_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        # 原始图像路径
        src_path = os.path.join(dataset_path, filename)
        
        # 目标路径
        base_name = os.path.splitext(filename)[0]
        dest_image_path = os.path.join(output_dataset_path, filename)
        dest_text_path = os.path.join(output_dataset_path, f"{base_name}.txt")
        
        try:
            # 复制图像到新目录
            img = Image.open(src_path).convert('RGB')
            img.save(dest_image_path)
            
            # 生成caption
            inputs = processor(
                images=img,
                return_tensors="pt"
            ).to(model.device, torch.float16)
            
            generated_ids = model.generate(**inputs, max_new_tokens=50)
            caption = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()
            
            # 保存文本标注
            with open(dest_text_path, 'w') as f:
                f.write(caption)
            
            print(f"Generated: {dest_image_path} -> {caption}")

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")