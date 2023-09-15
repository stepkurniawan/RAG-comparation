from langchain.document_loaders import WebBaseLoader, PyPDFLoader, DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS, Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np

FAISS_PATH = "vectorstores/db_faiss"
CHROMA_PATH = "vectorstores/db_chroma"

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

# create vector database
def create_local_faiss_vector_database(data, embeddings, DB_PATH):
    """
    Create a local vector database from a list of texts and an embedding model.
    """
    # Loader for PDFs
    # loader = DirectoryLoader(DATA_PATH, glob = '*.pdf', loader_cls= PyPDFLoader)
    # documents = loader.load()
    # text_splitter = RecursiveCharacterTextSplitter (chunk_size = 500, chunk_overlap = 50)
    # texts = text_splitter.split_documents(documents)

    texts = dataset_to_texts(data)
    text_splitter = RecursiveCharacterTextSplitter (chunk_size = 500, chunk_overlap = 50)
    texts = text_splitter.split_texts(texts)

    db = FAISS.from_texts(texts, embeddings)
    db.save_local(DB_PATH)


def get_faiss_vector_database(data, embeddings):
    
    db = FAISS.from_texts(data, embeddings)
    return db
    
def create_chroma_db(documents, embeddings):
    vectorstore = Chroma.from_documents(documents=documents, 
                                        embedding=embeddings,
                                        persist_directory=CHROMA_PATH)
    return vectorstore

def load_chroma_db(embeddings):
    vectorstore = Chroma(persist_directory=CHROMA_PATH, 
                         embedding_function=embeddings)
    return vectorstore
    

def similarity_search_doc(db, query):
    """
    Ref:
    https://github.com/JayZeeDesign/Knowledgebase-embedding/blob/main/app.py
    """
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    # print(page_contents_array)

    return page_contents_array