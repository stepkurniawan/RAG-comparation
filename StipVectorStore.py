from langchain.vectorstores import FAISS, Chroma
import time
from LogSetup import logger
from rag_splitter import split_data_to_docs


VECTORSTORE_NAMES = ['faiss', 'chroma']
VECTORSTORE_OBJS = [FAISS, Chroma]

FAISS_PATH = "vectorstores/db_faiss"
CHROMA_PATH = "vectorstores/db_chroma"



class StipVectorStore:
    def __init__(self, vectorstore_name):
        self.vectorstore_name = vectorstore_name # faiss or chroma
        self.docs_source = "" # source of the documents (suswiki, wikipedia, etc)
        self.embedding = None # embedding object
        self.model_name = "" # model name, taken from embedding
        self.vectorstore_with_model_path = "" # path to vectorstore with model name
        self.total_time = 0 # how long does it take in seconds to create vectorstore
        self.chunk_size = 0 # how many characters in a chunk
        self.chunk_overlap_scale = 0 # how much overlap between chunks
        self.k = 0 # number of nearest neighbors
        self.db = None # vectorstore object

        if vectorstore_name not in VECTORSTORE_NAMES:
            raise ValueError(f"vectorstore_name must be one of {VECTORSTORE_NAMES}")
        elif vectorstore_name == 'faiss':
            self.vectorstore_obj = FAISS 
            self.vectorstore_path = FAISS_PATH
        elif vectorstore_name == 'chroma':
            self.vectorstore_obj = Chroma
            self.vectorstore_path = CHROMA_PATH
        
    def create_vectorstore(self, dict_docs, embedding, chunk_size=200, chunk_overlap_scale=0.1):
        # check if the dict_docs is already documents, or is my dictionary that contain sources
        self.embedding = embedding
        self.model_name = self.embedding.model_name.split('/')[-1]
        self.chunk_size = chunk_size
        self.chunk_overlap_scale = chunk_overlap_scale

        # get source and the documents
        if type(dict_docs) == list:
            documents = dict_docs
            self.docs_source = None
        else:
            self.docs_source = dict_docs['source'] if 'source' in dict_docs.keys() else None
            documents = dict_docs['documents'] if 'documents' in dict_docs.keys() else dict_docs
        self.vectorstore_with_model_path = self.vectorstore_path+ "/" + self.docs_source + "/" + self.model_name + "_" + str(self.chunk_size) + "_" + str(self.chunk_overlap_scale)

        # splitting documents
        split_docs = split_data_to_docs(documents, chunk_size, chunk_overlap_scale)['documents']

        start_time = time.time()

        if self.vectorstore_name == 'faiss':
            self.db = self.vectorstore_obj.from_documents(split_docs, self.embedding)
            self.db.save_local(self.vectorstore_with_model_path)
            
        elif self.vectorstore_name == 'chroma':
            self.db = self.vectorstore_obj.from_documents(documents=split_docs,
                                            embedding=self.embedding,
                                            persist_directory=self.vectorstore_with_model_path)
        
        end_time = time.time()
        self.total_time = end_time-start_time

        print(f'success create vectorstore: {self.vectorstore_name} using {self.embedding} in {self.total_time} seconds')
        logger.info(f'success create vectorstore: {self.vectorstore_name} using {self.embedding} in {self.total_time} seconds')
        
        
        return self.db
    
    def load_vectorstore(self):
        start_time = time.time()
        db = None

        if self.vectorstore_name == 'faiss':
            db = self.vectorstore_obj.load_local(self.vectorstore_with_model_path, self.embedding)
        elif self.vectorstore_name == 'chroma':
            db = self.vectorstore_obj(persist_directory=self.vectorstore_with_model_path, 
                         embedding_function=self.embedding)
            
        end_time = time.time()
        print(f'success load vectorstore: {self.vectorstore_name} in {end_time-start_time} seconds')
        logger.info(f'success load vectorstore: {self.vectorstore_name} in {end_time-start_time} seconds')

        return db
