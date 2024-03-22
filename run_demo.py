from langchain.retrievers import TFIDFRetriever
from functools import lru_cache
import pandas as pd
from llama_cpp import Llama
from loguru import logger
from transformers import pipeline
import scipy
import time
import gradio as gr
from prompt_template import PROMPT_TEMPLATE

DOCUMENTATION_PATH = ".data/"

LLAMA_CPP_MODEL_PATH = "./models/zephyr-7b-beta.Q5_K_M.gguf"


def main(instructions: str, history: list):
    start_time = time.time()
    response = generate_response(instructions)
    logger.info(f"Model Response: {response}")
    end_time = time.time()
    runtime = round(end_time - start_time, 2)
    return response


def generate_response(instructions: str, history: list=[]):
    retriever = load_sparse_vector_db()
    relevant_documents = retriever.get_relevant_documents(instructions)
    prompt = PROMPT_TEMPLATE.format(articles=relevant_documents, instructions=instructions)
    logger.info(f"PROMPT: {prompt}")
    model = load_model(LLAMA_CPP_MODEL_PATH)
    raw_response = model(
        prompt,
        stream=True,
        temperature=0.6,
        repeat_penalty=1.2,
        max_tokens=200,
    )
    partial_message = ""
    for chunk in raw_response:
        if len(chunk['choices'][0]['text']) != 0:
            partial_message = partial_message + chunk['choices'][0]['text']
            yield partial_message


@lru_cache(maxsize=None)
def load_sparse_vector_db() -> TFIDFRetriever:
    """Load the sparse vector database"""
    dataset = pd.read_csv(DOCUMENTATION_PATH)
    # Concatenate each row of the dataframe into a single string
    dataset_list = dataset.to_dict("records")
    dataset_list = [str(row) for row in dataset_list]
    return TFIDFRetriever.from_texts(k=2, texts=dataset_list)


@lru_cache(maxsize=None)
def load_model(model_path: str):
    """Load Llama model"""
    return Llama(
        model_path=model_path,
        seed=32,
        n_gpu_layers=20,
        n_batch=512,
        n_ctx=12000,
    )


demo = gr.ChatInterface(
    generate_response,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Was ist Ihre Frage?", container=False, scale=7),
    title="Aktion Tier",
    description="Sprechen Sie mit Lana, unserem KI-Avatar",
    theme="soft",
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
)

if __name__ == "__main__":
    demo.launch(share=True)