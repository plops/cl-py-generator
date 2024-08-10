# https://huggingface.co/docs/diffusers/en/using-diffusers/img2img
# pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
# pip install diffusers transformers accelerate

import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid
torch.cuda.get_device_name(0)
pipeline = AutoPipelineForImage2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16, use_safetensors=True
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
#pipeline.enable_xformers_memory_efficient_attention()

init_image = load_image("IMG_0693.jpg") #"https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")
#prompt = "wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
prompt = "person, portrait, face, hair style, hair color, hair cut"
# IMPORTANT: Convert the prompt to float32
#text_input_ids = pipeline.tokenizer(prompt, return_tensors="pt").input_ids.to(pipeline.device).float()

# Run the image2image pipeline
#image = pipeline(prompt=prompt, image=init_image, text_input_ids=text_input_ids).images[0]
image = pipeline(prompt, image=init_image, guidance_scale=8.0).images[0]
# save image to png file
image.save("output.png")
  # make_image_grid([init_image, image], rows=1, cols=2)
