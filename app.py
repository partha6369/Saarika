# app.py

# === Install required libraries ===
import subprocess
import sys

def install_if_missing(pip_package, import_name=None):
    import_name = import_name or pip_package
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", pip_package],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )


# Attempt to install only if not present
install_if_missing("datasets")
install_if_missing("transformers")
install_if_missing("gradio")
install_if_missing("nltk")

import os
import random
import gradio as gr
from datasets import load_dataset
from transformers import pipeline

import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')

# Load the IMDB dataset
dataset = load_dataset("imdb")
all_reviews = dataset["train"]["text"] + dataset["test"]["text"]

# Load summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Clean and finalize the summary output
def clean_summary_text(summary):
    parts = summary.split('"')
    cleaned_sentences = []
    for part in parts:
        part = part.strip()
        if len(part) > 1:
            part = part[0].upper() + part[1:]
            if part[-1] not in '.!?':
                part += '.'
            cleaned_sentences.append(part)
    return ' '.join(cleaned_sentences)

def trim_summary_to_words(summary, max_words=60):
    sentences = sent_tokenize(summary)
    output = []
    word_count = 0

    for sent in sentences:
        words = sent.split()
        if word_count + len(words) <= max_words:
            output.append(sent)
            word_count += len(words)
        else:
            break
    return ' '.join(output)

def get_summary_length_bounds(text, max_words=60):
    avg_token_per_word = 1.3
    token_limit = int(max_words * avg_token_per_word)

    # Minimum should be about 40–50% of max
    max_length = min(token_limit, 120)      # Cap to model's safe upper range
    min_length = max(20, int(0.5 * max_length))  # Avoid too short summaries

    return max_length, min_length

# Summarisation function with word limit
def summarize_review(text, max_words=60):
    input_text = text[:1024]  # Truncate to fit model limits
    max_len, min_len = get_summary_length_bounds(input_text, max_words=max_words)
    summary = summarizer(
        input_text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False
    )[0]['summary_text']
    return trim_summary_to_words(clean_summary_text(summary), max_words=60)

# Get a random review and its summary
def random_review():
    review = random.choice(all_reviews)
    summary = summarize_review(review)
    return review, summary

# Summarize user input
def user_summary_fn(text, word_limit):
    summary = summarize_review(text, int(word_limit))
    return summary

# Try to load the PayPal URL from the environment; if missing, use a placeholder
paypal_url = os.getenv("PAYPAL_URL", "https://www.paypal.com/donate/dummy-link")

# Gradio Interface
with gr.Blocks(title="Saarika: Essence Extractor for Text") as demo:
    gr.Markdown("## 🪷 *Saarika* (सारिका) – Essence Extractor for Text using Transformers")

    gr.Markdown("### 🔁 Random IMDB Review and Summary")

    with gr.Row():
#         gr.Markdown("### 🔁 Random IMDB Review and Summary")

        # Get initial random review
        init_review, init_summary = random_review()

        review_display = gr.Textbox(label="Random IMDB Review", lines=10, interactive=False, value=init_review)
        summary_display = gr.Textbox(label="Generated Summary", lines=10, interactive=False, value=init_summary)

    refresh_btn = gr.Button("🔄 Try another")
    refresh_btn.click(fn=random_review, outputs=[review_display, summary_display])

    gr.Markdown("### 🧪 Summarise Your Own Text")
    with gr.Row():
        user_input = gr.Textbox(label="Enter your text here", lines=10,
                                placeholder="Paste or write your text...")
        word_limit = gr.Number(label="Summary word limit", value=60, precision=0)
    with gr.Row():
        summary_output = gr.Textbox(label="Generated Summary", lines=10, interactive=False)
    with gr.Row():
        submit_btn = gr.Button("🚀 Submit")
        clear_btn = gr.Button("🧹 Clear")

    submit_btn.click(fn=user_summary_fn, inputs=[user_input, word_limit], outputs=summary_output)
    clear_btn.click(
        fn=lambda: ("", 60, ""),
        inputs=[],
        outputs=[user_input, word_limit, summary_output]
    )

    with gr.Row():
        gr.HTML(f"""
        <a href="{paypal_url}" target="_blank">
            <button style="background-color:#0070ba;color:white;border:none;padding:10px 20px;
            font-size:16px;border-radius:5px;cursor:pointer;margin-top:10px;">
                ❤️ Support Research via PayPal
            </button>
        </a>
        """)

if __name__ == "__main__":
    # Determine if running on Hugging Face Spaces
    on_spaces = os.environ.get("SPACE_ID") is not None

    # Launch the app conditionally
    demo.launch(share=not on_spaces)
