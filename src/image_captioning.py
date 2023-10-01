################################################################################
# SECTION: Image Captioning
# DESCRIPTION: Image Captioning is the task of generating a textual description
# of an image. The model is trained to generate a caption given an image.
# DATE: Oct 1st 2023
################################################################################

import gradio as gr
from transformers import pipeline
from PIL import Image
import numpy as np
import io
import base64 

class ImageCaptioning:
    def __init__(self):
        # Initialize the pipeline for image captioning
        self.pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        # Load the 'success.png' image for comparison
        self.success_image = Image.open("example_images/success.png")

    def image_to_base64_str(self, pil_image):
        byte_arr = io.BytesIO()
        pil_image.save(byte_arr, format='PNG')
        byte_arr = byte_arr.getvalue()
        return str(base64.b64encode(byte_arr).decode('utf-8'))

    def images_equal(self, img1, img2):
        return np.array_equal(np.array(img1), np.array(img2))

    def captioner(self, image):
        # Check if the uploaded image matches 'success.png'
        if self.images_equal(image, self.success_image):
            return "What your organization will achieve by hiring Daniel Efting :)"

        # Convert image to base64 string
        base64_image = self.image_to_base64_str(image)
        
        # Get the caption
        result = self.pipeline(base64_image)
        return result[0]['generated_text']

    def interface(self):
        # Set up Gradio Interface
        return gr.Interface(
            fn=self.captioner,
            inputs=[gr.Image(label="Upload image", type="pil")],
            outputs=[gr.Textbox(label="Caption")],
            title="Image Captioning with BLIP",
            description="Caption any image using the BLIP model",
            allow_flagging="never",
            examples=["example_images/success.png", "example_images/cartoon.png"]
        )
