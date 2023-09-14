from langchain.llms import CTransformers # to use CPU only
from langchain.chat_models import ChatOpenAI
import os
import openai
from dotenv import load_dotenv
import os

def load_llm_ctra_llama27b():
    """
    loading CTransformers model
    """
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.0

    )
    return llm

def load_llm_gpt35():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(
        openai_api_key=openai.api_key, 
        model="gpt-3.5-turbo",
        temperature=0.0,
        )
    return llm