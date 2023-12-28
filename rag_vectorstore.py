from langchain.document_loaders import WebBaseLoader, PyPDFLoader, DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS, Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import SVMRetriever

from rag_load_data import load_sustainability_wiki_langchain_documents
from rag_splitter import split_data_to_docs
from rag_embedding import get_embed_model, embedding_ids

from datasets import Dataset, DatasetDict
import numpy as np

VECTORSTORE_NAMES = ['faiss', 'chroma']
VECTORSTORE_OBJS = [FAISS, Chroma]

FAISS_PATH = "vectorstores/db_faiss/"
CHROMA_PATH = "vectorstores/db_chroma/"




def get_index_vectorstore_wiki_nyc(embed_model):
    # load the Wikipedia page and create index
    loader = WebBaseLoader("https://en.wikipedia.org/wiki/New_York_City") # pip install bs4
    index = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,
        embedding=embed_model,
        # text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        # vectorstore_kwargs={ "persist_directory": "/vectorstore"},
    ).from_loaders([loader]) 
    return index


def dataset_to_texts(data):
    data_pd = data.to_pandas()
    texts = data_pd['chunk'].to_numpy()
    return texts

# deprecated #########################
def create_faiss_db(documents, embedding):    
    db = FAISS.from_documents(documents, embedding)
    db.save_local(FAISS_PATH+"_"+embedding.model_name )
    return db

def load_local_faiss_vector_database(embeddings):
    """
    Load a local vector database from a list of texts and an embedding model.
    """
    db = FAISS.load_local(FAISS_PATH, embeddings)
    return db


def create_chroma_db(documents, embeddings):
    vectorstore = Chroma.from_documents(documents, 
                                        embeddings,
                                        persist_directory=CHROMA_PATH)
    return vectorstore

def load_chroma_db(embeddings):
    vectorstore = Chroma(persist_directory=CHROMA_PATH, 
                         embedding_function=embeddings)
    return vectorstore
#####################################################################

def create_db_pipeline(knowledge_base, vectorstore_name, embed_id, chunk_size, chunk_overlap_scale):
    """
    Create a local vector database from a list of texts and an embedding model.
    you can change the input from texts or data
    """
    embedding_name = embed_id.split('/')[-1]

    # Load data 
    if knowledge_base == 'suswiki': 
        documents = load_sustainability_wiki_langchain_documents()
    elif knowledge_base == 'wikipedia':
        print("# TODO : get docu from wikipedia")
        
    # split docs
    split_docs = split_data_to_docs(documents, chunk_size, chunk_overlap_scale)

    # vectorstore
    embed_model , _ = get_embed_model(embed_id)
    if vectorstore_name=='FAISS':
        # db = FAISS.from_texts(texts, embeddings)
        save_path = FAISS_PATH + "/"  + "/" + embedding_name + "_" + str(chunk_size) + "_" + str(chunk_overlap_scale)
        db = FAISS.from_documents(split_docs, embed_model)
        db.save_local(save_path)
    elif vectorstore_name=='Chroma':
        save_path = CHROMA_PATH + "/" + "/" + embedding_name + "_" + str(chunk_size) + "_" + str(chunk_overlap_scale)
        db = Chroma.from_documents(documents=split_docs, 
                                            embedding=embed_model,
                                            persist_directory=save_path)
    
    return db

def load_db_pipeline(knowledge_base, vectorstore_name, embedding):
    """
    load db from local into variable
    """
    embedding_name = embedding.model_name.split('/')[-1]

    if vectorstore_name=='FAISS':
        load_path = FAISS_PATH + "/" + knowledge_base + "/" + embedding_name
        db = FAISS.load_local(load_path)

    if vectorstore_name=='Chroma':
        load_path = CHROMA_PATH + "/" + knowledge_base + "/" + embedding_name
        db = Chroma(persist_directory=load_path, 
                         embedding_function=embedding)
    return db




def similarity_search_doc(db, query, top_k=3):
    """
    Ref:
    https://github.com/JayZeeDesign/Knowledgebase-embedding/blob/main/app.py
    """
    similar_response = db.similarity_search(query, k=top_k)

    page_contents_array = get_content_from_similarity_search(similar_response)
    print(page_contents_array)

    # print(len(page_contents_array))

    return page_contents_array

def multi_similarity_search_doc(db, dataset, top_k=3):
    """
    Ref: similar to similarity_search_doc(), but for multiple queries
    input: dataset with columns: questions, ground_truths
    output: add new column: contexts
    """
    
    try: 
        dataset = dataset['train']
    except:
        pass

    # change dataset into dataframe
    dataframe = dataset.to_pandas()

    # for every question, do similarity search, save the result as a new column called "contexts"
    dataframe['contexts'] = dataframe['question'].apply(lambda q: similarity_search_doc(db, q, top_k))

    # save the dataframe as a new dataset
    dataset_new = Dataset.from_pandas(dataframe)
    
    return dataset_new


def svm_similarity_search_doc(documents, query, embed_model, top_k):

    svm_retriever = SVMRetriever.from_documents(documents=documents,
                                                embeddings=embed_model,
                                                k = top_k,
                                                relevancy_threshold = 0.3)
    
    docs_svm=svm_retriever.get_relevant_documents(query)
    docs_svm_list = get_content_from_similarity_search(docs_svm)
    len(docs_svm)
    return docs_svm_list

def get_content_from_similarity_search(similar_response):
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array

