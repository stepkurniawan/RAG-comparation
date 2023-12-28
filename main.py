#%% 0. Import libraries
# !pip install bs4 chromadb tiktoken faiss-cpu accelerate xformers ragas

import os
from rag_load_data import get_arxiv_data_from_dataset, load_from_webpage
from rag_embedding import get_embed_model
from rag_embedding import embedding_ids
from rag_vectorstore import create_faiss_db, load_local_faiss_vector_database
from rag_vectorstore import get_index_vectorstore_wiki_nyc, create_chroma_db, load_chroma_db
from rag_vectorstore import svm_similarity_search_doc, similarity_search_doc
from rag_llms import load_llm_ctra_llama27b, load_llm_gpt35, load_llm_tokenizer_hf_with_model
from rag_llms import LLAMA2_13B_CHAT_MODEL_ID
# from rag_prompting import set_custom_prompt, set_custom_prompt_new, get_formatted_prompt
from rag_chains import retrieval_qa_chain_from_local_db, chain_with_docs, final_result
from rag_ragas import make_eval_chains, evaluate_RAGAS
from rag_splitter import split_data_to_docs
from StipKnowledgeBase import StipKnowledgeBase

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
# data = load_from_webpage("https://sustainabilitymethods.org/index.php/A_matter_of_probability")
suswiki_kb = StipKnowledgeBase("stepkurniawan/sustainability-methods-wiki", None, "huggingface_cache/suswiki_hf")
# wikipedia_kb = KnowledgeBase("wikipedia", "20220301.simple", "huggingface_cache/wikipedia_hf")
data = suswiki_kb.load_documents()
print("the data is: ", data)

# 3.1.1 limit data for test
data = data[:5]

# 3.2 Split text into chunks ##############
docs = split_data_to_docs(data=data, chunk_size=200, chunk_overlap_scale=0.1)

# 3.3 embedding ###########
# embed_model = get_retriever_embeddings()
embed_model = get_embed_model(embedding_ids['HF_BER_ID_2'])

# 4 VECTOR STORE ######################################################
## create LOCAL FAISS
#%% # if folder DB_FAISS_PATH is empty, then run 
# if len(os.listdir(DB_PATH)) == 0:
# create_local_faiss_vector_database(texts=docs, embeddings=embed_model, DB_PATH=DB_PATH) # maybe for dataset ? 
create_chroma_db(docs, embed_model) 


## 4.4 index
# index = rag_vectorstore.get_index_vectorstore_wiki_nyc(embed_model)

## 4.5 Similiarity search
db = load_local_faiss_vector_database(embed_model)
# db = load_chroma_db(embed_model)

# similar_response = similarity_search_doc(db, QUERY)
similar_docs = svm_similarity_search_doc(docs, QUERY, embed_model, 1)


### 4.5 EVALUATE RETRIEVER : context precision , recall, and F-measure
retriever_evaluation = evaluate(
                    dataset,
                    metrics=[
                            context_precision,
                            faithfulness,
                            answer_relevancy,
                            context_recall,
                            # harmfulness,
                        ],
                        )


#%% 5. GENERATOR #####################################################
## 5.1 embedding
# the embedding of the generator is already inside the model

## 5.3 prompt
# prompt = get_formatted_prompt(context=similar_docs, query=QUERY) # TODO: check if this is correct

## 5.4 LLM model : Select by comment and uncommenting the code below 
# llm = load_llm_ctra_llama27b() 
# llm = load_llm_gpt35()
llm = load_llm_tokenizer_hf_with_model(LLAMA2_13B_CHAT_MODEL_ID) # note: it works using worker22

print("success loading llm model")

## 5.5 Chain
# the similar resopnse is not used here
# but the chain is summarizing itself from the vector store, based on the arguments there
qa_chain = retrieval_qa_chain_from_local_db(llm=llm, vectorstore=db) 

## 5.6 Result
qa_chain_result = final_result(qa_chain, QUERY)
# qa_chain_result = chain_with_docs(qa_chain, similar_response, QUERY) # deprecated...

print("the result is: ", qa_chain_result['result'])
print("the source documents are: ", qa_chain_result['source_documents'])

#%% 6. EVALUATION ########################################################
## preparing RAGAS table
"""
ref: https://github.com/explodinggradients/ragas
"""

#### TODO: clean up
# eval_df = evaluate_RAGAS(qa_chain_result)
# print(f"{qa_chain_result=}")




## LLM model