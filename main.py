#%% 0. Import libraries
import os
from rag_embedding import get_retriever_embeddings, get_generator_embeddings
from rag_knowledge_base import get_arxiv_data_from_dataset
from rag_vectorstore import dataset_to_texts, create_local_faiss_vector_database
from rag_llms import load_llm_ctra_llama27b
from rag_prompting import set_custom_prompt
from rag_chains import retrieval_qa_chain_from_local_db, qa_bot
from langchain.vectorstores import FAISS

#%% 1. UI ##############################################################

#%% 2. Set up environment ############################################
query = "What is the best way to learn new subjects?"

#%% 3. RETRIEVER #####################################################
# 3.1 embedding
embed_model = get_retriever_embeddings()

# 3.2 knowledge base / dataset
data = get_arxiv_data_from_dataset()

# 3.2 Split text into chunks
# its part of loader

# 3.3 vectorstore

## LLAMA FAISS CODE
DB_FAISS_PATH = "vectorstores/db_faiss"
texts = dataset_to_texts(data)

#%% # if folder DB_FAISS_PATH is empty, then run 
if len(os.listdir(DB_FAISS_PATH)) == 0:
    create_local_faiss_vector_database(texts, embed_model, DB_FAISS_PATH) 

## RAGAS code
# index = rag_vectorstore.get_index_vectorstore_wiki_nyc(embed_model)

#%% 4. GENERATOR #####################################################
## 4.1 embedding
embeddings_gen = get_generator_embeddings()

## 4.2 Knowledge Base / DB
db = FAISS.load_local(DB_FAISS_PATH, embeddings_gen)

## 4.3 prompt
prompt = set_custom_prompt()

## 4.4 model
llm = load_llm_ctra_llama27b()

## 4.5 Chain
qa_chain = retrieval_qa_chain_from_local_db(llm=llm, prompt_template=prompt, db=db)

## 4.6 bot / agent
# qa_bot_pipe = qa_bot(db, llm, prompt)

## 4.7 Result
qa_chain_result = qa_chain({"query": query})
print(qa_chain_result)

#%% 5. EVALUATION ########################################################
## RAGAS criteria

## LLM model