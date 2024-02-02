#%%# prepare environement
import os
import time

import pandas as pd

from rag_vectorstore import multi_similarity_search_doc

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
QUESTION_DATASET = load_50_qa_dataset()['train']
FOLDER_PATH ="experiments/Embeddings/"

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

contexted_faiss_bge = multi_similarity_search_doc(faiss_bge, QUESTION_DATASET, TOP_K)
contexted_faiss_gte = multi_similarity_search_doc(faiss_gte, QUESTION_DATASET, TOP_K)
contexted_faiss_uae = multi_similarity_search_doc(faiss_uae, QUESTION_DATASET, TOP_K)

#%% Evaluation
## create evaluation FIRST TIME ONLY
# evaluated_bge = retriever_evaluation(contexted_faiss_bge, FOLDER_PATH+"bge_eval.csv")
# evaluated_gte = retriever_evaluation(contexted_faiss_gte, FOLDER_PATH+"gte_eval.csv")
# evaluated_uae = retriever_evaluation(contexted_faiss_uae, FOLDER_PATH+"uae_eval.csv")

## load evaluation
evaluated_bge = pd.read_csv(FOLDER_PATH+"bge_eval.csv")
evaluated_gte = pd.read_csv(FOLDER_PATH+"gte_eval.csv")
evaluated_uae = pd.read_csv(FOLDER_PATH+"uae_eval.csv")


#%% Calculate mean score
mean_precision_bge = evaluated_bge['context_precision'].mean()
mean_recall_bge = evaluated_bge['context_recall'].mean()
f1_bge = 2* mean_precision_bge * mean_recall_bge / (mean_precision_bge + mean_recall_bge)

mean_precision_gte = evaluated_gte['context_precision'].mean()
mean_recall_gte = evaluated_gte['context_recall'].mean()
f1_gte  = 2* mean_precision_gte * mean_recall_gte / (mean_precision_gte + mean_recall_gte)

mean_precision_uae = evaluated_uae['context_precision'].mean()
mean_recall_uae = evaluated_uae['context_recall'].mean()
f1_uae = 2* mean_precision_uae * mean_recall_uae / (mean_precision_uae + mean_recall_uae)

## store them in a dataframe
result_df = pd.DataFrame({
    'mean_precision': [mean_precision_bge, mean_precision_gte, mean_precision_uae],
    'mean_recall': [mean_recall_bge, mean_recall_gte, mean_recall_uae],
    'f1': [f1_bge, f1_gte, f1_uae]
    }, index=['bge', 'gte', 'uae'])


print("")


# %%
