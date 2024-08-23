import gradio as gr

from transformers import pipeline

title = "Document Image Transformer"

description = "Gradio Demo for DiT, the Document Image Transformer pre-trained on IIT-CDIP, a dataset that includes 42 million document images and fine-tuned on RVL-CDIP, a dataset consisting of 400,000 grayscale images in 16 classes, with 25,000 images per class. To use it, simply add your image, or click one of the examples to load them. Read more at the links below."

article = "<p style='text-align: center'><a href='https://huggingface.co/microsoft/dit-base-finetuned-rvlcdip' target='_blank'>Huggingface Model</a></p>"

pipe = pipeline(
    task="image-classification",
    model="microsoft/dit-base-finetuned-rvlcdip"
)

gr.Interface.from_pipeline(
    pipe,
    title=title,
    description=description,
    examples=[
        'coca_cola_advertisement.png',
        'scientific_publication.png',
        'letter.jpeg'
    ],
    article=article,
    enable_queue=True,
).launch()