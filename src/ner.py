################################################################################
# SECTION: Named Entity Recognition (NER)
# DESCRIPTION: Named Entity Recognition (NER) is the task of classifying words in
# a text into pre-defined categories. The categories can be of the form of
# person, location, organization, etc.
# DATE: Oct 1st 2023
################################################################################
import gradio as gr
from transformers import pipeline


class NamedEntityRecognition:
    def __init__(self, model="dslim/bert-base-NER"):
        self.pipeline = pipeline("ner", model=model)
        self.examples = [
            "My name is Dan! I come from the Midwest! Who's the man? Dan! Who's got a plan? Dan! Who can? Dan can! Shibooyah! Shibooyah! Shibooyah, yeah, yeah!",
            "Apple Inc. is an American multinational technology company headquartered in Cupertino, California.",
            "Barack Obama was born on August 4, 1961, in Honolulu, Hawaii.",
            "The Amazon rainforest produces more than 20% of the world's oxygen.",
            "The Mona Lisa is a 16th century oil painting created by Leonardo.",
            "In 2020, Tesla's stock price surged, making Elon Musk one of the richest people in the world.",
            "Bitcoin, a digital currency, reached an all-time high value in 2021.",
            "The Great Wall of China is one of the Seven Wonders of the World.",
            "In Greek mythology, Zeus is the king of the gods, and he lives on Mount Olympus.",
            "Wimbledon is the oldest tennis tournament in the world, and it was first played in 1877."
        ]

    def ner(self, input_text):
        return self.pipeline(input_text)

    def interface(self):
        return gr.Interface(
            fn=self.ner,
            inputs=[gr.Textbox(label="Text to find entities", lines=2, placeholder="Enter text here...")],
            outputs=[gr.HighlightedText(label="Text with entities")],
            title="Named Entity Recognition with BERT",
            description="Identify and categorize entities in a given text using NER.",
            allow_flagging="never",
            examples=self.examples
        )
