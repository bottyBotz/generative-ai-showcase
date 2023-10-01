#Load your api key and relevant python libraries
import os
from IPython.display import Image, display, HTML
import IPython.display
from PIL import Image
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']
import gradio as gr
from transformers import pipeline
import requests, json
from diffusers import DiffusionPipeline



from src.text_summarization import TextSummarization
from src.ner import NamedEntityRecognition
from src.image_captioning import ImageCaptioning


################################################################################
# SECTION: Text to Image (Stable Diffusion)
# DESCRIPTION: Stable Diffusion is a generative technique for generating images from
# text. 
# DATE: Oct 1st 2023
################################################################################


diffuser_pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")

def get_diffuser_completion(prompt, diffuser_pipeline):
    return diffuser_pipeline(prompt).images[0]    

def generate_diffused_img(prompt):
    img = get_diffuser_completion(prompt)
    return img


# def image_generation_interface():
#     with gr.Blocks() as demo:
#         gr.Markdown("# Image Generation with Stable Diffusion")
#         with gr.Row():
#             with gr.Column(scale=4):
#                 prompt = gr.Textbox(label="Your prompt") #Give prompt some real estate
#             with gr.Column(scale=1, min_width=50):
#                 btn = gr.Button("Submit") #Submit button side by side!
#         with gr.Accordion("Advanced options", open=False): #Let's hide the advanced options!
#                 negative_prompt = gr.Textbox(label="Negative prompt")
#                 with gr.Row():
#                     with gr.Column():
#                         steps = gr.Slider(label="Inference Steps", minimum=1, maximum=100, value=25,
#                         info="In many steps will the denoiser denoise the image?")
#                         guidance = gr.Slider(label="Guidance Scale", minimum=1, maximum=20, value=7,
#                         info="Controls how much the text prompt influences the result")
#                     with gr.Column():
#                         width = gr.Slider(label="Width", minimum=64, maximum=512, step=64, value=512)
#                         height = gr.Slider(label="Height", minimum=64, maximum=512, step=64, value=512)
#         output = gr.Image(label="Result") #Move the output up too
                
#         btn.click(fn=generate_diffused_img), inputs=[prompt,negative_prompt,steps,guidance,width,height], outputs=[output])

################################################################################
# SECTION: Gradio App
# DESCRIPTION: Build the Gradio App using the interfaces defined above.
# DATE: Oct 1st 2023
################################################################################

class GradioApp:
    def __init__(self):
        gr.close_all() # Close all existing Gradio apps
        self.ner = NamedEntityRecognition()
        self.img_captioning = ImageCaptioning()
        self.text_summ = TextSummarization()
        #self.img_gen = ImageGeneration()

    def build_app(self, share=False):
        with gr.Blocks() as demo:
            gr.Markdown("All in one Generative AI Showcase.")
            with gr.Tab("Text Summarization"):
                self.text_summ.interface()
            with gr.Tab("Named Entity Recognition (NER)"): 
                self.ner.interface()
            with gr.Tab("Image Captioning"):
                self.img_captioning.interface()
            #with gr.Tab("Image Generation"):
            #    pass
            #image_generation_interface()
            # # Image Generation!
            # get_completion = pipeline("image-to-image-ca", model="valhalla/image-to-image-agent-clevr")
            # gr.Interface(fn=get_completion,
            #     inputs=[gr.Image(label="Input Image")],
            #     outputs=[gr.Image(label="Generated Image")],
            #     title="Image Generation with valhalla/image-to-image-agent-clevr",
            #     description="Generate any image using the `valhalla/image-to-image-agent-clevr` model under the hood!")
            # Launch the Gradio App
            demo.launch(share=share)


def main():
    app = GradioApp()
    app.build_app(share=True)
    
if __name__ == '__main__':
    main()