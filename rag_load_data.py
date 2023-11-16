from datasets import load_dataset, Dataset
from langchain.document_loaders import WebBaseLoader, HuggingFaceDatasetLoader
from langchain.document_loaders import JSONLoader


import os
import pandas as pd


WIKI_PATH = "data/Sustainability+Methods_dump.xml"
JSON_PATH = "data/Sustainability+Methods_dump.xml.json"

HF_HUB_QA_DATASET = "stepkurniawan/qa_sustainability_wiki"
HF_HUB_QA_DATASET_2 = "stepkurniawan/sustainability-methods-wiki"
HF_HUB_QA_LLAMA = "stepkurniawan/qa-rag-llama"
HF_HUB_RAGAS = "stepkurniawan/RAGAS_50"

QA_GT_JSON_PATH = "data/collection_ground_truth_ragas_chatgpt4.json"

KNOWLEDGE_BASE_IDS = ['suswiki', 'wikipedia']


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
    # output : DatasetDict (from hugging face)
    data = load_dataset("wikipedia", "20220301.simple", split='train[:10000]')
    print(f'success load data: {data}')
    return data


#### SUSTAINABILITY WIKI #### 
#### we have 2 options to load the data:
def load_sustainability_wiki_dataset():
    # its in hf_dataset format, just like get_wikipedia_data_from_dataset
    dataset_name = "stepkurniawan/sustainability-methods-wiki"
    dataset_from_hf = load_dataset(dataset_name, split='train')
    print(f'success load data from huggingface: stepkurniawan/sustainability-methods-wiki')
    return dataset_from_hf

def load_sustainability_wiki_langchain_documents():
    # the result is Document(page_content=...) just like load_from_webpage()
    dataset_name = "stepkurniawan/sustainability-methods-wiki"
    page_content_column = "text"

    loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)
    data = loader.load()
    print(f'success load data from huggingface: stepkurniawan/sustainability-methods-wiki')
    return data



def load_from_webpage(link):
    loader = WebBaseLoader(link, verify_ssl=False)
    data = loader.load()
    print(f'success load data from webpage: {link}')
    return data

def get_qa_dataframe():
    """
    get question answer dataframe 
    REF: RAGAS quickstart https://colab.research.google.com/github/explodinggradients/ragas/blob/main/docs/quickstart.ipynb#scrollTo=f17bcf9d
    the dataset must consists of: 
    1. question
    2. ((answer)) -> this will be populated later, when the RAG model answers the question
    3. contexts
    4. ground_truths
    but in this case, we only consider 1 context and 1 ground_truth
    """

    loader = JSONLoader(
    file_path=QA_GT_JSON_PATH,
    jq_schema='.messages[].content',
    text_content=False)


    qa_gt_df = loader.load() # type: Documents

    return qa_gt_df


# %% take 3 examples from these questions 
def load_qa_rag_dataset():
    # load from HF
    dataset = load_dataset(HF_HUB_QA_DATASET)
    print("success loading question answer dataset from HF")
    print("the first question is: ", dataset['train']['question'][0])
    print("the first ground truth is: ", dataset['train']['ground_truths'][0])
    return dataset

# qa_dataset = load_qa_rag_dataset()
# take the first 3 rows from the "train" dataset
# qa_dataset = qa_dataset['train'][:3]

def load_50_qa_dataset():
    dataset = load_dataset(HF_HUB_QA_DATASET_2, "50_QA")
    print("success loading question answer dataset from HF")
    dataset = dataset.select_columns(['question', 'ground_truths'])     # drop dataset['train']['contexts'] and dataset['train']['summary'] because we will use retriever to fill that

    print("the first question is: ", dataset['train']['question'][0])
    print("the first ground truth is: ", dataset['train']['ground_truths'][0])
    return dataset