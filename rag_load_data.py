from datasets import load_dataset, Dataset
from langchain.document_loaders import WebBaseLoader
import os

import xmltodict
import json

import pandas as pd
import html


WIKI_PATH = "data/Sustainability+Methods_dump.xml"
JSON_PATH = "data/Sustainability+Methods_dump.xml.json"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

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

def create_json_from_xml(xml_path):
    with open(xml_path, 'r', encoding='utf-8') as f:
        data_dict = xmltodict.parse(f.read())
        json_data = json.dumps(data_dict, indent=4)
        
    with open(xml_path+".json", "w") as json_file:
        json_file.write(json_data)
    
def load_dict_from_json(json_path):
    with open(json_path, 'r') as f:
        data_dict = json.load(f)
    return data_dict

def create_dataset_to_current_directory():
    """
    ref: https://huggingface.co/docs/datasets/v2.14.5/en/package_reference/main_classes#datasets.Dataset.from_dict
    """
    ds = load_dataset("rotten_tomatoes", split="validation")
    dir_path = os.path.join(CURRENT_DIR, "data")
    ds.info.write_to_directory(dir_path)
    return ds

def load_dataframe_from_json(json_path):
    dict = load_dict_from_json(json_path)
    pages = dict['mediawiki']['page']
    df = pd.DataFrame(pages)
    df['text'] = df['revision'].apply(lambda x: html.unescape(x['text']['#text']))
    df.drop(columns=['revision', 'ns', 'id'], inplace=True)

    print("success load dataframe from json: ", json_path)

    return df


