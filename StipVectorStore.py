from langchain.vectorstores import FAISS, Chroma
import time
from LogSetup import logger
from rag_splitter import split_data_to_docs
import pickle


VECTORSTORE_NAMES = ['faiss', 'chroma']
VECTORSTORE_OBJS = [FAISS, Chroma]

FAISS_PATH = "vectorstores/db_faiss"
CHROMA_PATH = "vectorstores/db_chroma"



class StipVectorStore:
    def __init__(self, vectorstore_name):
        self.vectorstore_name = vectorstore_name # faiss or chroma
        self.docs_source = "" # source of the documents (suswiki, wikipedia, etc)
        self.embedding = None # embedding object
        self.embedding_name = "" # embedding name, taken from embedding
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



    def load_vectorstore(self, vectorstore_path):
            start_time = time.time()

            # Load the class attributes
            with open(vectorstore_path + '/_attributes.pkl', 'rb') as f:
                self.__dict__.update(pickle.load(f))

            if self.vectorstore_name == 'faiss':
                self.db = self.vectorstore_obj.load_local(self.vectorstore_with_model_path, self.embedding)
            elif self.vectorstore_name == 'chroma':
                self.db = self.vectorstore_obj(persist_directory=self.vectorstore_with_model_path, 
                            embedding_function=self.embedding)
                
            end_time = time.time()
            print(f'success load vectorstore: {self.vectorstore_name} in {end_time-start_time} seconds')
            logger.info(f'success load vectorstore: {self.vectorstore_name} in {end_time-start_time} seconds')

            return self.db




        
    def create_vectorstore(self, dict_docs, embedding, chunk_size=200, chunk_overlap_scale=0.1):
        # check if the dict_docs is already documents, or is my dictionary that contain sources
        self.embedding = embedding
        self.embedding_name = self.embedding.model_name.split('/')[-1]
        self.chunk_size = chunk_size
        self.chunk_overlap_scale = chunk_overlap_scale

        # get source and the documents
        if type(dict_docs) == list:
            documents = dict_docs
            self.docs_source = None
        else:
            self.docs_source = dict_docs['source'] if 'source' in dict_docs.keys() else None
            documents = dict_docs['documents'] if 'documents' in dict_docs.keys() else dict_docs
        self.vectorstore_with_model_path = self.vectorstore_path+ "/" + self.docs_source + "/" + self.embedding_name + "_" + str(self.chunk_size) + "_" + str(self.chunk_overlap_scale)

        # splitting documents
        split_docs = split_data_to_docs(documents, chunk_size, chunk_overlap_scale)['documents']
        
        start_time = time.time()

        if self.vectorstore_name == 'faiss':
            try:
                print(f'!NOTE: start from_documents(): {self.vectorstore_name}')
                self.db = self.vectorstore_obj.from_documents(split_docs, self.embedding)
                print(f'!NOTE: success create vectorstore using {self.embedding_name}: {self.vectorstore_name}')
                self.db.save_local(self.vectorstore_with_model_path)
                print(f'!NOTE: success save vectorstore: {self.vectorstore_name} in {self.vectorstore_with_model_path}')
            except Exception as e:
                print(f"!NOTE: Exception occurred while creating FAISS vectorstore using {self.embedding_name}: {e}")

    
        elif self.vectorstore_name == 'chroma':
            try:
                print(f'!NOTE: start from_documents(): {self.vectorstore_name}')
                self.db = self.vectorstore_obj.from_documents(documents=split_docs,
                                                embedding=self.embedding,
                                                persist_directory=self.vectorstore_with_model_path)
                print(f'!NOTE: success save vectorstore: {self.vectorstore_name} in {self.vectorstore_with_model_path}')

            except Exception as e:
                print(f"!NOTE: Exception occurred while creating CHROMA vectorstore {self.vectorstore_name} using {self.embedding_name}: {e}")


        end_time = time.time()
        self.total_time = end_time-start_time
        print(f'!NOTE: success create vectorstore: {self.vectorstore_name} in {self.total_time} seconds')


        try:
        # save metadata
            with open(self.vectorstore_with_model_path + "/_attributes.pkl", "wb") as f:
                
                # not storing: db
                pickle.dump({
                    'vectorstore_name': self.vectorstore_name,
                    'docs_source': self.docs_source,
                    'embedding': self.embedding,
                    'embedding_name': self.embedding_name,
                    'vectorstore_with_model_path': self.vectorstore_with_model_path,
                    'total_time': self.total_time,
                    'chunk_size': self.chunk_size,
                    'chunk_overlap_scale': self.chunk_overlap_scale,
                    'k': self.k
                }, f, pickle.HIGHEST_PROTOCOL)

                # save all metadata, with the embedding its 1.2 GB, with db its 1.3 GB
                # self.db = None
                # pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
                # self.db = self.load_vectorstore(self.vectorstore_with_model_path)

        except Exception as e:
            print(f"Exception occurred while saving metadata: {e}")

        print(f'success create vectorstore: {self.vectorstore_name} using {self.embedding_name} in {self.total_time} seconds')
        logger.info(f'success create vectorstore: {self.vectorstore_name} using {self.embedding_name} in {self.total_time} seconds')
        
        
        return self.db
    
    
