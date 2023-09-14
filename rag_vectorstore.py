from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np




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
def create_local_faiss_vector_database(texts, embeddings, DB_PATH):
    """
    Create a local vector database from a list of texts and an embedding model.
    """
    # Loader for PDFs
    # loader = DirectoryLoader(DATA_PATH, glob = '*.pdf', loader_cls= PyPDFLoader)
    # documents = loader.load()
    # text_splitter = RecursiveCharacterTextSplitter (chunk_size = 500, chunk_overlap = 50)
    # texts = text_splitter.split_documents(documents)

    text_splitter = RecursiveCharacterTextSplitter (chunk_size = 500, chunk_overlap = 50)
    texts = text_splitter.split_texts(texts)

    db = FAISS.from_texts(texts, embeddings)
    db.save_local(DB_PATH)

