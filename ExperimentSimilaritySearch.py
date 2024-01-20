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

#### 
EMBED_MODEL = StipEmbedding("bge").embed_model
CHUNK_SIZE = 200
CHUNK_OVERLAP_SCALE = 0.1


#### Knowledge Base 
suswiki_kb = StipKnowledgeBase("stepkurniawan/sustainability-methods-wiki", None, "huggingface_cache/suswiki_hf")
suswiki_docs = suswiki_kb.load_documents()
# suswiki_docs["documents"] = suswiki_docs["documents"][:100] # limit data for test
print(f'successful load data from {suswiki_docs["source"]}. It has source and documents keys')

#### Embedding

#### Split and create vectorstore

EUCLEDIAN = "l2"
COSINE = "cosine"
INNER_PRODUCT = "ip"

## 1. faiss
faiss_vs = StipVectorStore("faiss")
# faiss_eucledian = faiss_vs.create_vectorstore(suswiki_docs, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP_SCALE, EUCLEDIAN)
# faiss_cosine = faiss_vs.create_vectorstore(suswiki_docs, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP_SCALE, COSINE)
# faiss_ip = faiss_vs.create_vectorstore(suswiki_docs, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP_SCALE, INNER_PRODUCT)


# ## 2. chroma
chroma_vs = StipVectorStore("chroma")
# chroma_eucledian = chroma_vs.create_vectorstore(suswiki_docs, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP_SCALE, EUCLEDIAN)
chroma_cosine = chroma_vs.create_vectorstore(suswiki_docs, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP_SCALE, COSINE)
# chroma_ip = chroma_vs.create_vectorstore(suswiki_docs, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP_SCALE, INNER_PRODUCT)


#### Load vectorstore
# faiss_vs.load_vectorstore("vectorstores/db_faiss/sustainability-methods-wiki/bge-large-en-v1.5_200_0.1_l2")
chroma_vs.load_vectorstore("vectorstores/db_chroma/sustainability-methods-wiki/bge-large-en-v1.5_200_0.1_cosine")
#%% 


