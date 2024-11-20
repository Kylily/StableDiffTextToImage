import tkinter as tk
import customtkinter as ctk

from PIL import ImageTk
import torch
from diffusers import StableDiffusionPipeline

# Create app
app = ctk.CTk()
app.geometry("532x622")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(master=app, height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white")
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(master=app, height=512, width=512)
lmain.place(x=10, y=110)

# Use CPU bc Macbook M2
modelid = "CompVis/stable-diffusion-v1-4"
device = 'cpu'
pipe = StableDiffusionPipeline.from_pretrained(modelid)
pipe.to(device)

def generate():
    # Generate image using the pipeline
    image = pipe(prompt.get(), guidance_scale=8.5).images[0]
    image.save('generatedimage.png')
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img)
    lmain.image = img 

trigger = ctk.CTkButton(master=app, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", command=generate)
trigger.configure(text='Generate')
trigger.place(x=206, y=60)

app.mainloop()
