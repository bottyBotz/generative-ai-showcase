from transformers import pipeline
import gradio as gr

################################################################################
# SECTION: Text Summarization
# DESCRIPTION: Text Summarization is the task of generating a textual summary
# of a given text. The model is trained to generate a summary given a text.
# DATE: Oct 1st 2023
################################################################################

# class TextSummarization:
#     def __init__(self, model = "facebook/bart-large-cnn"):
#         # Initialize the pipeline for text summarization
#         self.pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
#         self.examples = [
#             # Example about Eiffel Tower
#             "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.",
            
#             # Example from Scientific Article
#             "Recent studies have shown that climate change poses a significant threat to coral reefs and marine biodiversity. Elevated sea temperatures, ocean acidification, and increased frequency of extreme weather events have caused widespread coral bleaching, habitat degradation, and a decline in fish populations. Coral reefs are vital ecosystems that provide a habitat for a quarter of all marine species and are crucial for the livelihoods of millions of people around the world. The destruction of coral reefs will have devastating consequences for marine biodiversity and human communities that rely on them. Current efforts to mitigate the effects of climate change on coral reefs include the establishment of marine protected areas, coral farming, and the development of heat-resistant coral strains. However, these measures are not sufficient to counteract the long-term effects of climate change. There is an urgent need for coordinated global action to reduce greenhouse gas emissions and protect our oceans for future generations.",
            
#             # Example from News Article
#             "In the last decade, the automotive industry has seen a dramatic shift towards electric vehicles (EVs). With the increasing awareness of climate change and the need for sustainable transportation, consumers are more inclined to opt for electric cars. Major car manufacturers like Tesla, Ford, and Nissan have entered the market with a range of electric models, and governments worldwide are offering incentives to encourage EV adoption. However, challenges remain. The infrastructure for electric charging stations is still not widespread, and there are concerns about the long-term viability of batteries. Nonetheless, experts predict that by 2030, electric cars will account for more than 50% of new vehicle sales, signaling a transformative change in how we think about transportation. While electric cars are not a panacea for all environmental issues, they represent a significant step forward in reducing carbon emissions and combating climate change.",
            
#             # Example from Literature
#             "Modern literature often delves into the intricacies of human existence. Characters are crafted to be relatable, flawed, and multi-dimensional, reflecting the complexities of real-life individuals. Themes often revolve around existential questions, moral dilemmas, and the search for meaning. One notable example is the novel 'The Unbearable Lightness of Being' by Milan Kundera, which explores the tension between fate and free will through its characters' lives and choices. The book challenges conventional notions of love, identity, and morality, making readers question their own beliefs and actions. Such works contribute to a broader cultural dialogue about what it means to be human in a world full of uncertainty and change."
#         ]

#     def summarize_text(self, input_text):
#         # Generate summary using the pipeline
#         output = self.pipeline(input_text)
#         return output[0]['summary_text']

#     def interface(self):
#         # Set up Gradio Interface
#         return gr.Interface(
#             fn=self.summarize_text,
#             inputs=[gr.Textbox(label="Text to summarize", lines=6)],
#             outputs=[gr.Textbox(label="Summary", lines=3)],
#             title="Text Summarization with Facebook's Large BART model",
#             description="Summarize any text using the `facebook/bart-large-cnn` model under the hood!",
#             examples=self.examples
#         )


class TextSummarization:
    def __init__(self):
        # Initialize the pipeline for text summarization
        self.pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
        self.examples = {
            "Science Text": "Example science text here...",
            "News Article": "Example news article here...",
            "Literature": "Example literature text here..."
        }

    def summarize_text(self, input_text, summary_length=100):
        # Generate summary using the pipeline
        output = self.pipeline(input_text, max_length=summary_length)
        return output[0]['summary_text']

    def load_example(self, example_key):
        return self.examples.get(example_key, "")

    def interface(self):
        with gr.Blocks() as app:
            gr.Markdown("## Instantly Turn Long Texts into Short, Informative Summaries!")
            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(label="Text to summarize", lines=6, placeholder="Paste your lengthy text here...")
                with gr.Column():
                    example_dropdown = gr.Dropdown(label="Or try an example", choices=list(self.examples.keys()), type="value")
                    load_example_btn = gr.Button(label="Load Example")
            with gr.Accordion("Advanced Settings", open=False):
                summary_length = gr.Slider(label="Summary Length", minimum=10, maximum=200)
            
            summarize_btn = gr.Button(label="Summarize!")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Original Text")
                    original_text = gr.Textbox(lines=3, disabled=True)
                with gr.Column():
                    gr.Markdown("### Summary")
                    summary_text = gr.Textbox(lines=3, disabled=True)
                    
            gr.Markdown("Got another article to summarize? Go ahead, paste it above!")

        return app, {
            "btn_summarize": summarize_btn.click(
                fn=self.summarize_text, 
                inputs=[input_text, summary_length], 
                outputs=[summary_text]
            ),
            "btn_load_example": load_example_btn.click(
                fn=self.load_example,
                inputs=[example_dropdown],
                outputs=[input_text]
            )
        }
