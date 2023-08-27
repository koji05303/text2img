import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
from diffusers import StableDiffusionPipeline

from colorama import init
from termcolor import cprint 
from termcolor import colored
from pyfiglet import figlet_format

######Title######
cprint(figlet_format('TEXT2IMG', font = 'starwars'),
	   'white', 'on_red', attrs = ['bold'])

######Import Model######
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16) ###VRAM太少把浮點數精度限止在16位元而不是32
pipe = pipe.to(device)

while True:

    ######Enter Sentenses######
    prompt = input(colored("Enter your sentence here (type 'quit' to exit): ", 'green'))

    ######Check quit byte######
    if prompt.lower() == "quit":
        break

    ######Generate & Save######
    image = pipe(prompt).images[0]
    file_name = prompt + ".png"
    output_path = "/home/wei/coding/text2img/saved_img/"
    image.save(os.path.join(output_path, file_name))
    print("Image saved in:", os.path.join(output_path, file_name))


######Exit######
cprint(figlet_format('EXIT!', font = 'starwars'),
	   'black', 'on_blue', attrs = ['bold'])


######Enter Sentenses######
#prompt = input(colored("Enter ur sentense here : ",'green'))
#image = pipe(prompt).images[0]  

######File Path######
#file_name = prompt + ".png"

#image.save(os.path.join(output_path,file_name))
#print("image save in : ",output_path,file_name)
