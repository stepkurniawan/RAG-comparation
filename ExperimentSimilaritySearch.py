"""
Experiment to compare different Similarity Search Algorithm and see the effect on the retriever performance
The similarity search algorithm are:
1. Eucledian Distance
2. Cosine Similarity
3. SVM
"""
# compare with 1 knowledge base first, and then compare with 2 knowledge bases, 
# if there is no significant difference, then we can use 1 knowledge base only in the paper

#%% 0. Import libraries
# !pip install bs4 chromadb tiktoken faiss-cpu accelerate xformers ragas

import os
import time

from rag_vectorstore import similarity_search_doc, multi_similarity_search_doc

from LogSetup import logger
from rag_ragas import retriever_evaluation

from StipVectorStore import StipVectorStore
from StipKnowledgeBase import StipKnowledgeBase, StipHuggingFaceDataset
from StipEmbedding import StipEmbedding

#### Knowledge Base 
suswiki_kb = StipKnowledgeBase("stepkurniawan/sustainability-methods-wiki", None, "huggingface_cache/suswiki_hf")
suswiki_docs = suswiki_kb.load_documents()
print(f'successful load data from {suswiki_docs["source"]}. It has source and documents keys')

#### Split and create vectorstore
faiss_vs = StipVectorStore("faiss")
# faiss_sus_bge_db = faiss_vs.create_vectorstore(suswiki_docs, StipEmbedding("bge").embed_model, chunk_size=200, chunk_overlap_scale=0.1)

#### Load vectorstore
faiss_vs.load_vectorstore("vectorstores/db_faiss/sustainability-methods-wiki/bge-large-en-v1.5_200_0.1")

#%% 


