# app.py

import gradio as gr
from chatbot import get_faq_response
from transformers import pipeline
from typing import Union

# Load speech recognition pipeline
speech_to_text = pipeline("automatic-speech-recognition", model="openai/whisper-base")

def combined_input_handler(audio_file: Union[str, None], text_input: str) -> str:
    if audio_file:
        result = speech_to_text(audio_file)
        query = result["text"]
    elif text_input:
        query = text_input
    else:
        return "Please ask a question using voice or text."

    return get_faq_response(query)

iface = gr.Interface(
    fn=combined_input_handler,
    inputs=[
        gr.Audio(source="microphone", type="filepath", label="🎤 Ask by voice"),
        gr.Textbox(label="⌨️ Or type your question")
    ],
    outputs="markdown",
    title="🛍️ E-commerce FAQ Chatbot",
    description="Ask anything about orders, shipping, returns, payments, etc. Use voice or text."
)

if __name__ == "__main__":
    iface.launch()