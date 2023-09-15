#%% 0. Import libraries
# !pip install bs4 chromadb tiktoken faiss-cpu

import os
from rag_embedding import get_retriever_embeddings, get_generator_embeddings
from rag_load_data import get_arxiv_data_from_dataset, load_from_webpage
from rag_vectorstore import dataset_to_texts, create_local_faiss_vector_database, similarity_search_doc
from rag_llms import load_llm_ctra_llama27b, load_llm_gpt35
from rag_prompting import set_custom_prompt, set_custom_prompt_new, get_formatted_prompt
from rag_chains import retrieval_qa_chain_from_local_db, qa_bot
from rag_ragas import make_eval_chains, evaluate_RAGAS

from langchain.vectorstores import FAISS
from langchain.llms import CTransformers # to use CPU only

from ragas.metrics import faithfulness, answer_relevancy, context_relevancy, context_recall
from ragas.langchain import RagasEvaluatorChain

import pandas as pd


#%% 1. UI ##############################################################

#%% 2. Set up environment ############################################
QUERY = "What is the probability of you being so much taller than the average? "
RAGAS_METRICS = [faithfulness, answer_relevancy, context_relevancy]
DB_PATH = "vectorstores/db_faiss"
LINK = "https://sustainabilitymethods.org/index.php/A_matter_of_probability"


#%% 3. RETRIEVER #####################################################

# 3.1 Load knowledge base / dataset #######
# data = get_arxiv_data_from_dataset()
data = load_from_webpage("https://sustainabilitymethods.org/index.php/A_matter_of_probability")

# 3.2 Split text into chunks ##############

# 3.1 embedding ############################
embed_model = get_retriever_embeddings()

# 3.3 vectorstore ###########################

## create LOCAL FAISS
#%% # if folder DB_FAISS_PATH is empty, then run 
# if len(os.listdir(DB_PATH)) == 0:
create_local_faiss_vector_database(data, embed_model, DB_PATH) 


## 3.4 index
# index = rag_vectorstore.get_index_vectorstore_wiki_nyc(embed_model)

## 3.5 Similiarity search
db = FAISS.load_local(DB_PATH, embed_model)
similar_response = similarity_search_doc(db, QUERY)

#%% 4. GENERATOR #####################################################
## 4.1 embedding
# the embedding of the generator is already inside the model

## 4.3 prompt
prompt = get_formatted_prompt(context=similar_response[0].page_content, query=QUERY)

## 4.4 LLM model : Select by comment and uncommenting the code below 
# llm = load_llm_ctra_llama27b() 
llm = load_llm_gpt35()

## 4.5 Chain
# qa_chain = retrieval_qa_chain_from_local_db(llm=llm, prompt_template=prompt, db=db)
qa_chain = prompt | llm | db.as_retriever(search_kwargs = {'k':3})

## 4.6 bot / agent
# qa_bot_pipe = qa_bot(db, llm, prompt)

## 4.7 Result
# qa_chain_result = qa_chain({"query": QUERY})
qa_chain_result = qa_chain.invoke({"query": QUERY})
print(qa_chain_result)

#%% 5. EVALUATION ########################################################
## RAGAS criteria
# RAGAS_METRICS = [faithfulness, answer_relevancy, context_relevancy, context_recall]
eval_df = pd.DataFrame(columns=['query', 'answer', 'context'] + [m.name for m in RAGAS_METRICS])

eval_chains = make_eval_chains(RAGAS_METRICS)

evaluate_RAGAS(eval_chains, qa_chain_result, eval_df)

print("")

## LLM model