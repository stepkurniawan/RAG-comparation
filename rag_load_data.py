from datasets import load_dataset, Dataset
from langchain.document_loaders import WebBaseLoader
import os


WIKI_PATH = "data/Sustainability+Methods_dump.xml"
JSON_PATH = "data/Sustainability+Methods_dump.xml.json"


def get_arxiv_data_from_dataset():
    """
    REF: https://github.com/pinecone-io/examples/blob/master/learn/generation/llm-field-guide/llama-2/llama-2-13b-retrievalqa.ipynb
    ex query: Explain to me the difference between nuclear fission and fusion.
    """
    data = load_dataset(
        'jamescalam/llama-2-arxiv-papers-chunked',
        split='train'
    )
    print(f'success load data: {data}')
    return data

def get_wikipedia_data_from_dataset():
    data = load_dataset("wikipedia", "20220301.simple", split='train[:10000]')
    print(f'success load data: {data}')
    return data

def load_sustainability_wiki_dataset():
    load_dataset_from_hf = load_dataset("stepkurniawan/sustainability-methods-wiki", split='train')
    print(f'success load data from huggingface: stepkurniawan/sustainability-methods-wiki')
    return load_dataset_from_hf


def load_from_webpage(link):
    loader = WebBaseLoader(link)
    data = loader.load()
    print(f'success load data from webpage: {link}')
    return data
