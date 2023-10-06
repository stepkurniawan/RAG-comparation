#%% 0. Import libraries
# !pip install bs4 chromadb tiktoken faiss-cpu accelerate xformers ragas

import os
from rag_load_data import get_arxiv_data_from_dataset, load_from_webpage, load_sustainability_wiki_dataset, load_sustainability_wiki_langchain_documents
from rag_embedding import get_retriever_embeddings, get_generator_embeddings
from rag_vectorstore import create_local_faiss_vector_database, load_local_faiss_vector_database
from rag_vectorstore import get_index_vectorstore_wiki_nyc, create_chroma_db, load_chroma_db
from rag_vectorstore import svm_similarity_search_doc, similarity_search_doc
from rag_llms import load_llm_ctra_llama27b, load_llm_gpt35, load_llm_tokenizer_llama2_13b_hf
from rag_prompting import set_custom_prompt, set_custom_prompt_new, get_formatted_prompt
from rag_chains import retrieval_qa_chain_from_local_db, chain_with_docs, final_result
from rag_ragas import make_eval_chains, evaluate_RAGAS
from rag_splitter import split_data_to_docs

from langchain.vectorstores import FAISS
from langchain.llms import CTransformers # to use CPU only

from ragas.langchain import RagasEvaluatorChain

import pandas as pd


#%% 1. UI ##############################################################

#%% 2. Set up environment ############################################
QUERY = "When and where did the quantitative Content Analysis method originate?"
DB_PATH = "vectorstores/db_faiss"
LINK = "https://sustainabilitymethods.org/index.php/A_matter_of_probability"

#%% 3. RETRIEVER #####################################################

# 3.1 Load knowledge base / dataset #######
# data = get_arxiv_data_from_dataset()
# data = load_from_webpage("https://sustainabilitymethods.org/index.php/A_matter_of_probability")
# data2 = load_sustainability_wiki_dataset()
data = load_sustainability_wiki_langchain_documents()
# print("the data is: ", data)

# 3.2 Split text into chunks ##############
docs = split_data_to_docs(data)

# 3.1 embedding ###########
embed_model = get_retriever_embeddings()

# 3.3 VECTOR STORE ######################################################

## create LOCAL FAISS
#%% # if folder DB_FAISS_PATH is empty, then run 
# if len(os.listdir(DB_PATH)) == 0:
# create_local_faiss_vector_database(texts=docs, embeddings=embed_model, DB_PATH=DB_PATH) # maybe for dataset ? 
# create_chroma_db(docs, embed_model) 


## 3.4 index
# index = rag_vectorstore.get_index_vectorstore_wiki_nyc(embed_model)

## 3.5 Similiarity search
db = load_local_faiss_vector_database(embed_model)
# db = load_chroma_db(embed_model)

# similar_response = similarity_search_doc(db, QUERY)
similar_docs = svm_similarity_search_doc(docs, QUERY, embed_model)

#%% 4. GENERATOR #####################################################
## 4.1 embedding
# the embedding of the generator is already inside the model

## 4.3 prompt
# prompt = get_formatted_prompt(context=similar_docs, query=QUERY) # TODO: check if this is correct

## 4.4 LLM model : Select by comment and uncommenting the code below 
# llm = load_llm_ctra_llama27b() 
llm = load_llm_gpt35()
# llm = load_llm_tokenizer_llama2_13b_hf() # TODO: not working because of model.layers.0.mlp.gate_proj.weight doesn't have any device set.

print("success loading llm model")

## 4.5 Chain
# the similar resopnse is not used here
# but the chain is summarizing itself from the vector store, based on the arguments there
qa_chain = retrieval_qa_chain_from_local_db(llm=llm, vectorstore=db) 

## 4.7 Result
qa_chain_result = final_result(qa_chain, QUERY)
# qa_chain_result = chain_with_docs(qa_chain, similar_response, QUERY) # deprecated...

print("the result is: ", qa_chain_result['result'])
print("the source documents are: ", qa_chain_result['source_documents'])

#%% 5. EVALUATION ########################################################
## preparing RAGAS table
"""
ref: https://github.com/explodinggradients/ragas
"""

#### TODO: clean up
# eval_df = evaluate_RAGAS(qa_chain_result)
# print(f"{qa_chain_result=}")




## LLM model