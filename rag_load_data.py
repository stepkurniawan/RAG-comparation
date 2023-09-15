from datasets import load_dataset
import tiktoken
from langchain.document_loaders import WebBaseLoader


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


def load_from_webpage(link):
    loader = WebBaseLoader(link)
    data = loader.load()
    print(f'success load data from webpage: {link}')
    return data

