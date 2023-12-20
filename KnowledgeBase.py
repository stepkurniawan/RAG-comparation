from datasets import load_dataset, Dataset
from langchain.schema import Document

from langchain.document_loaders import WebBaseLoader, HuggingFaceDatasetLoader
from langchain.document_loaders import JSONLoader, TextLoader
from typing import Iterator, List, Mapping, Optional, Sequence, Union


import os
import pandas as pd

########### KNOWLEDGE BASE & Questions Dataset ###########
class KnowledgeBase:
    def __init__(
            self, 
            hf_path:str,
            subset_name: Optional[str],
            cache_dir: Optional[str],
            ):
        """ Initialize the Knowledge Base
        Args:
            hf_path: Path or name of the dataset.
            config_name: Name of the dataset configuration.
            cache_dir: Directory to read/write data.
        """
        self.hf_path = hf_path
        self.subset_name = subset_name
        self.cache_dir = cache_dir

    
    # load data from suswiki
    def load_documents(self):
        # the result is Document(page_content=...) just like load_from_webpage()
        # self.path = "stepkurniawan/sustainability-methods-wiki"
        # usage ex: KnowledgeBase(WIKI_PATH).load_documents() -> langchain.Documents

        #### if path in cache_dir exists, then load Documents from cache_dir
        documents = []

        if os.path.exists(self.cache_dir):
            dataset = load_dataset(
                self.cache_dir,
            )
            hf_dataset = dataset['train']
            
            for entry in hf_dataset:
                doc = Document(page_content=entry['text'])
                documents.append(doc)
            
            print(f'success load data from cache_dir: {self.cache_dir}')
            return documents # Dataset ({feautres: ['title', 'text'], num_rows: 226})
        ########################
        else:
            loader = HuggingFaceDatasetLoader(
                path=self.hf_path, 
                name=self.subset_name,
                cache_dir=self.cache_dir,)
            documents = loader.load()
            print(f'success load data from huggingface: {self.hf_path}')

            return documents


class HuggingFaceDataset:
    def __init__(
            self, 
            hf_path):
        
        self.hf_path = hf_path

    def load_dataset(self):
        dataset_from_hf = load_dataset(self.hf_path, split='train')
        print(f'success load data from huggingface: {self.hf_path}')
        return dataset_from_hf

# decommisioned
WIKI_PATH = "data/Sustainability+Methods_dump.xml"
JSON_PATH = "data/Sustainability+Methods_dump.xml.json"

HF_HUB_QA_DATASET = "stepkurniawan/qa_sustainability_wiki" # 647 curated questions from the sustainability wiki dump (question, ground_truths)
HF_HUB_QA_DATASET_2 = "stepkurniawan/sustainability-methods-wiki" # contains the whole dump files of sustainability wiki. (title, text)
HF_HUB_QA_LLAMA = "stepkurniawan/qa-rag-llama" # 50 questions that has been answered by the RAG models, the subset is the name of the llm model
HF_HUB_RAGAS = "stepkurniawan/RAGAS_50" # placeholder for the new 50 questions

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
        text_content=False
        )


    qa_gt_df = loader.load() # type: Documents

    return qa_gt_df


# %% take 3 examples from these questions 
def load_qa_rag_dataset():
    # load from HF
    dataset = load_dataset(HF_HUB_QA_DATASET, None)
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


##########################################################################
# data = get_wikipedia_data_from_dataset()
# suswiki_kb = KnowledgeBase("stepkurniawan/sustainability-methods-wiki", None, "data/suswiki_hf")
# wikipedia_kb = KnowledgeBase("wikipedia", "20220301.simple", "data/wikipedia_hf")
# suswiki_document = suswiki_kb.load_documents()
# print(suswiki_document)
