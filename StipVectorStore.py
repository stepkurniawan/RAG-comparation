import shutil
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores.utils import (
    DistanceStrategy,
    maximal_marginal_relevance,
)

from typing import Iterator, List, Mapping, Optional, Sequence, Union


import chromadb
import os

import time
from LogSetup import logger
from rag_splitter import split_data_to_docs
import pickle


VECTORSTORE_NAMES = ['faiss', 'chroma']
VECTORSTORE_OBJS = [FAISS, Chroma]

FAISS_PATH = "vectorstores/db_faiss"
CHROMA_PATH = "vectorstores/db_chroma"



class StipVectorStore:
    def __init__(self, vectorstore_name, index_distance="l2"):
        self.vectorstore_name: str = vectorstore_name # faiss or chroma
        self.docs_source: str = "" # source of the documents (suswiki, wikipedia, etc)
        self.embedding = None # embedding object
        self.embedding_name: str = "" # embedding name, taken from embedding
        self.save_path: str = "" # path to vectorstore with model name
        self.total_time: float = 0 # how long does it take in seconds to create vectorstore
        self.chunk_size: int = 0 # how many characters in a chunk
        self.chunk_overlap_scale: float = 0 # how much overlap between chunks
        self.k: int = 0 # number of nearest neighbors
        self.db = None # vectorstore object
        self.index_distance: Optional[str] = index_distance  # default is euclidean 'l2', Inner product	'ip', Cosine similarity	'cosine'
        self.ndata: int = 0 # number of data in vectorstore

        # # Mapping dictionary for vector stores
        # vector_stores = {
        #     'faiss': (FAISS, FAISS_PATH),
        #     'chroma': (Chroma, CHROMA_PATH)
        # }

        # # Check if vectorstore_name is valid
        # if self.vectorstore_name not in vector_stores:
        #     raise ValueError(f"vectorstore_name must be one of {list(vector_stores.keys())}")

        # # Assign vectorstore_obj and vectorstore_path using tuple unpacking
        # self.vectorstore_obj, self.vectorstore_path = vector_stores[vectorstore_name]


        ### deprecated ####
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
                faiss_distance_strategy = get_faiss_distance_strategy(self.index_distance)
                self.db = self.vectorstore_obj.load_local(self.save_path, self.embedding, distance_strategy=faiss_distance_strategy)
            elif self.vectorstore_name == 'chroma':
                self.db = self.vectorstore_obj(persist_directory=self.save_path, 
                            embedding_function=self.embedding,
                            )
                print(f'!NOTE: hnsw space: {self.db._collection.metadata["hnsw:space"]}')
                
                
            end_time = time.time()
            print(f'success load vectorstore: {vectorstore_path} in {end_time-start_time} seconds')
            logger.info(f'success load vectorstore: {vectorstore_path} in {end_time-start_time} seconds')
            
            return self.db
    




        
    def create_vectorstore(self, 
                           dict_docs, 
                           embedding, 
                           chunk_size, 
                           chunk_overlap_scale, 
                           index_distance: Optional[str],
                            ):
        
        # set attributes
        self.embedding = embedding
        self.embedding_name = self.embedding.model_name.split('/')[-1]
        self.chunk_size = chunk_size
        self.chunk_overlap_scale = chunk_overlap_scale
        self.index_distance = index_distance

        
        # check if the dict_docs is already documents, or is my dictionary that contain sources
        # get source and the documents
        print("Checking type of dict_docs...")
        if type(dict_docs) == list:
            print("dict_docs is a list.")
            documents = dict_docs
            self.docs_source = None
        else:
            print("dict_docs is a dictionary.")
            self.docs_source = dict_docs['source'] if 'source' in dict_docs.keys() else None
            documents = dict_docs['documents'] if 'documents' in dict_docs.keys() else dict_docs
        print("Type of dict_docs checked.")

        self.save_path = self.vectorstore_path+ "/" + self.docs_source + "/" + self.embedding_name + "_" + str(self.chunk_size) + "_" + str(self.chunk_overlap_scale)+"_"+self.index_distance

        # splitting documents
        print("Splitting documents...")
        split_docs = split_data_to_docs(documents, chunk_size, chunk_overlap_scale)['documents']
        print("Documents split.")

        start_time = time.time()

        if self.vectorstore_name == 'faiss':
            """
            Creating FAISS Vector Store
            https://python.langchain.com/docs/integrations/vectorstores/faiss
            """
            print("Creating FAISS Vector Store...")
            faiss_distance_strategy = get_faiss_distance_strategy(self.index_distance)
            try:
                print("Trying to create vectorstore from documents...")
                self.db = self.vectorstore_obj.from_documents(split_docs, self.embedding, distance_strategy=faiss_distance_strategy)
                insanity_check = self.db.similarity_search_with_score("A/B testing", k=3)

                self.ndata = self.db.index.ntotal
                print(f'!NOTE: distance strategy: {self.db.distance_strategy}')
                print(f'!NOTE: success save vectorstore: {self.vectorstore_name} in {self.save_path}')
                print(f'!NOTE: how many datapoints in vectorstore: {self.db.index.ntotal}')
                logger.info(f'!NOTE: success save vectorstore: {self.vectorstore_name} in {self.save_path}')
                logger.info(f'!NOTE: how many datapoints in vectorstore: {self.db.index.ntotal}')
                
                print("Saving vectorstore locally...")
                self.db.save_local(self.save_path)
                print("Vectorstore saved.")
                
            except Exception as e:
                print(f"!NOTE: Exception occurred while creating FAISS vectorstore using {self.embedding_name}: {e}")

    


        elif self.vectorstore_name == 'chroma':
            """
            Creating Chroma Vector Store
            https://python.langchain.com/docs/integrations/vectorstores/chroma#similarity-search-with-score
            """
            #### USING CHROMA NATIVE COLLECTION ####
            ### pass a chroma client to into Langchain : ref: https://python.langchain.com/docs/integrations/vectorstores/chroma 
            # persistent_client = chromadb.PersistentClient(path="./" + self.save_path)
            # collection_name = self.docs_source+"_"+self.embedding_name+"_"+str(self.chunk_size)+"_"+str(self.chunk_overlap_scale)+"_"+str(self.k)+"_"+index_distance

            # # delete if the collection already exist, if not create a new collection
            # try: 
            #     collection = persistent_client.get_collection(name=collection_name)
            #     persistent_client.delete_collection(name=collection_name)
            # except ValueError:
            #     # First time creating the collection
            #     print(f'!NOTE: First time creating the collection: {collection_name}')

            # collection = persistent_client.create_collection(name=collection_name,
            #                                                     metadata={"hnsw:space": self.index_distance})# default index_distance is euclidean 'l2', Inner product 'ip', Cosine similarity 'cosine'
            
            # print(f'!NOTE: here, the collection count should be 0: {collection.count()}')

            # # add documents to the collection
            # ids_list = list(map(str, range(len(split_docs)))) # manually create ids
            # page_contents = [doc.page_content for doc in split_docs]
            # # page_contents_short = [doc.page_content for doc in split_docs if len(doc.page_content) <= 100] # DEBUG only: to check if there is any too short empty page_content
            # collection.add(ids=ids_list, documents=page_contents, embeddings=self.embedding) # TODO this throws error because embeddings should be a list, so i should have len(split_docs) amount of embeeding
            # print(f'!NOTE: here, the collection count should be {len(split_docs)}: {collection.count()}')
            # self.ndata = collection.count()

            # # create langchain chroma client replaces from_documents()
            # self.db = Chroma(
            #     client=persistent_client,
            #     collection_name=collection_name,
            #     embedding_function=self.embedding,
            #     )

            
            ####  deprecated: using langchain's from_documents() #####
            # if there is already folder in self.save_path, delete it, otherwise continue
            if os.path.exists(self.save_path):
                shutil.rmtree(self.save_path)
                print(f'!NOTE: delete existing folder: {self.save_path}')
                print(f'!NOTE: start from_documents(): {self.vectorstore_name}')
            else:
                print(f'!NOTE: start from_documents(): {self.vectorstore_name}')

            self.db = self.vectorstore_obj.from_documents(documents=split_docs,
                                            embedding=self.embedding,
                                            persist_directory=self.save_path,
                                            collection_metadata ={"hnsw:space": self.index_distance}) # default is euclidean 'l2', Inner product	'ip', Cosine similarity	'cosine'
            
            print(f'!NOTE: success save vectorstore: {self.vectorstore_name} in {self.save_path}')
            print(f'!NOTE: hnsw space: {self.db._collection.metadata["hnsw:space"]}')
            print(f'!NOTE: how many datapoints in vectorstore - should be len(split_docs): {self.db._collection.count()}')

        end_time = time.time()
        self.total_time = end_time-start_time

        try:
        # save metadata
            with open(self.save_path + "/_attributes.pkl", "wb") as f:
                obj_dict = self.__dict__.copy()  # Create a copy of the object's attribute dictionary
                obj_dict.pop('db', None)  # Remove the 'db' attribute
                pickle.dump(obj_dict, f, pickle.HIGHEST_PROTOCOL)  # Pickle the modified dictionary
        except Exception as e:
            print(f"Exception occurred while saving metadata: {e}")

        print(f'!NOTE: success create vectorstore: {self.vectorstore_name} using {self.embedding_name} in {self.total_time} seconds or {self.total_time/60} minutes')
        logger.info(f'!NOTE: success create vectorstore: {self.vectorstore_name} using {self.embedding_name} in {self.total_time} seconds or {self.total_time/60} minutes')
        
        
        return self.db
    

def get_faiss_distance_strategy(index_distance):
    # Mapping dictionary for distance strategies
    distance_strategies = {
        "ip": DistanceStrategy.MAX_INNER_PRODUCT,
        "cosine": DistanceStrategy.COSINE
    }
    # Default to EUCLIDEAN_DISTANCE if no match found in the dictionary
    faiss_distance_strategy = distance_strategies.get(index_distance, DistanceStrategy.EUCLIDEAN_DISTANCE)
    return faiss_distance_strategy