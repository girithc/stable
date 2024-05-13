import requests
import torch
from PIL import Image
from io import BytesIO
from transformers import CLIPTokenizer, CLIPTextModel

from diffusers import StableDiffusionImg2ImgPipeline



# device = "mps"
# model_id_or_path = "runwayml/stable-diffusion-v1-5"
# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
# pipe = pipe.to(device)




# Initialize the tokenizer and text encoder for CLIP
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# Initialize and load the Stable Diffusion pipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    use_auth_token=True  # If you're using the Hugging Face Model Hub
).to("mps")  # Use CPU for inference


url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

# response = requests.get(url)
init_image = Image.open("tom.png").convert("RGB")
init_image = init_image.resize((768, 512))

prompt = "Man with sunglasses."

current_image = init_image

output = pipe(prompt=prompt, image=current_image, strength=0.8, guidance_scale=7.5)
current_image = output.images[0]

# Save the final image
current_image.save("fantasy_landscape.png")