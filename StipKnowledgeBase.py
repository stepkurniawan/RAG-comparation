from datasets import load_dataset, Dataset
from langchain.schema import Document

from langchain_community.document_loaders import WebBaseLoader, HuggingFaceDatasetLoader
from langchain_community.document_loaders import JSONLoader, TextLoader
from typing import Iterator, List, Mapping, Optional, Sequence, Union

import time
import os
import pandas as pd

HUGGINGFACE_CACHE_PATH = "huggingface_cache/"

########### KNOWLEDGE BASE & Questions Dataset ###########
class StipKnowledgeBase:
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
    def load_documents(self, limit: Optional[int] = None):
        """
        Load documents from huggingface or cache_dir based on StipKnowledgeBase object

        input: StipKnowledgeBase object
        output: ```
            dict_documents = {
                source: suswiki / wikipedia
                documents: list of Documents
                }
                ```
        """
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
            
        ########################
        else:
            loader = HuggingFaceDatasetLoader(
                path=self.hf_path, 
                name=self.subset_name,
                cache_dir=self.cache_dir,)
            documents = loader.load()
            print(f'success load data from huggingface: {self.hf_path}')

        # add the source to the document
        dict_documents = {
            "source": self.hf_path.split("/")[-1],
            "documents": documents[:limit]
        }
        return dict_documents
    
def load_suswiki():
    suswiki_kb = StipKnowledgeBase("stepkurniawan/sustainability-methods-wiki", None, "huggingface_cache/suswiki_hf")
    suswiki_docs = suswiki_kb.load_documents()
    return suswiki_docs

def load_wikipedia():
    wikipedia_kb = StipKnowledgeBase("wikipedia", "20220301.simple", "huggingface_cache/wikipedia_hf")
    wikipedia_docs = wikipedia_kb.load_documents()
    return wikipedia_docs


class StipHuggingFaceDataset:
    def __init__(
            self, 
            hf_path:str,
            subset:str,
            select_columns: Optional[List[str]] = None,):
        
        self.hf_path = hf_path
        self.subset = subset
        self.select_columns = select_columns

    def load_dataset(self):
        start_time = time.time()
        dataset_from_hf = load_dataset(self.hf_path, self.subset, download_mode="force_redownload")
        dataset_from_hf = dataset_from_hf.select_columns(self.select_columns)
        end_time = time.time()
        total_time = end_time - start_time
        print(f"total time to load dataset from HF: {total_time:.2f} seconds")
        print(f'success load data from huggingface: {self.hf_path}')
        print(f'The dataset has {dataset_from_hf.num_rows} rows and {len(dataset_from_hf.column_names)} columns')
        print(f'dataset_from_hf["train"]["question"][0]: {dataset_from_hf["train"]["question"][0]}')
        print(f'dataset_from_hf["train"]["ground_truths"][0]: {dataset_from_hf["train"]["ground_truths"][0]}')
        return dataset_from_hf

def load_50_qa_dataset():
    # dataset = load_dataset("stepkurniawan/sustainability-methods-wiki", "50_QA_reviewed", download_mode="force_redownload")
    dataset = load_dataset("stepkurniawan/sustainability-methods-wiki", "50_QA_reviewed")

    dataset = dataset.select_columns(['question', 'ground_truths']) 
    print("success loading question ground_truth dataset from HF")
    return dataset
    
    


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


##########################################################################
# data = get_arxiv_data_from_dataset()
# data = get_wikipedia_data_from_dataset()
# suswiki_kb = KnowledgeBase("stepkurniawan/sustainability-methods-wiki", None, "data/suswiki_hf")
# wikipedia_kb = KnowledgeBase("wikipedia", "20220301.simple", "data/wikipedia_hf")
# suswiki_document = suswiki_kb.load_documents()
# print(suswiki_document)


