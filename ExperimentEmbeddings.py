# prepare environement
import os
import time

import pandas as pd

from rag_vectorstore import similarity_search_doc, multi_similarity_search_doc

from LogSetup import logger
from rag_ragas import retriever_evaluation

from StipVectorStore import StipVectorStore
from StipKnowledgeBase import StipKnowledgeBase, StipHuggingFaceDataset , load_suswiki, load_wikipedia, load_50_qa_dataset
from StipEmbedding import StipEmbedding


#### fixed
CHUNK_SIZE = 200
CHUNK_OVERLAP_SCALE = 0.1
TOP_K = 3
INDEX_DISTANCE = "l2"
VECTORSTORE = StipVectorStore("faiss")
QUESTION_DATASET = StipHuggingFaceDataset().load_50_qa_dataset()['train']

suswiki_kb = load_suswiki()
wikipedia_kb = load_wikipedia()

#%% PREPARE VECTOR STORE
# faiss_bge = VECTORSTORE.create_vectorstore(KNOWLEDGE_BASE_SUSWIKI, StipEmbedding("bge").embed_model, CHUNK_SIZE, CHUNK_OVERLAP_SCALE, INDEX_DISTANCE)
# faiss_gte = VECTORSTORE.create_vectorstore(KNOWLEDGE_BASE_SUSWIKI, StipEmbedding("gte").embed_model, CHUNK_SIZE, CHUNK_OVERLAP_SCALE, INDEX_DISTANCE)
# faiss_uae = VECTORSTORE.create_vectorstore(KNOWLEDGE_BASE_SUSWIKI, StipEmbedding("uae").embed_model, CHUNK_SIZE, CHUNK_OVERLAP_SCALE, INDEX_DISTANCE)

#### Load VectorStore
faiss_bge = VECTORSTORE.load_vectorstore("vectorstores/db_faiss/sustainability-methods-wiki/bge-large-en-v1.5_200_0.1_l2")
faiss_gte = VECTORSTORE.load_vectorstore("vectorstores/db_faiss/sustainability-methods-wiki/gte-large_200_0.1_l2")
faiss_uae = VECTORSTORE.load_vectorstore("vectorstores/db_faiss/sustainability-methods-wiki/UAE-Large-V1_200_0.1_l2")



#%% Retrieval 
