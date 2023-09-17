#%%
from rag_embedding import get_retriever_embeddings, get_generator_embeddings
from rag_prompting import set_custom_prompt
from rag_ragas import evaluate_RAGAS
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from rag_llms import load_llm_ctra_llama27b, load_llm_gpt35
from langchain.llms import CTransformers # to use CPU only
from rag_chains import retrieval_qa_chain_from_local_db, final_result
from langchain import PromptTemplate
from langchain.chains import RetrievalQA

import json
from fastapi.encoders import jsonable_encoder


QUERY = "What is the probability of you being so much taller than the average? "

DB_FAISS_PATH = "vectorstores/db_faiss"


#%%
def test_get_generator_embeddings():
    embed_model = get_generator_embeddings()
    assert isinstance(embed_model.model_name, str) , "Failed getting the embedding model, check get_generator_embeddings()"
    print(f'Embedding model name: {embed_model.model_name}')
    return embed_model

test_get_generator_embeddings()

#%%

def test_read_db():
    db = FAISS.load_local("vectorstores/db_faiss", get_generator_embeddings())
    assert isinstance(db, FAISS) , "Failed getting the db, check test_read_db()"
    return db

test_db = test_read_db()

#%%
def test_set_custom_prompt():
    prompt = set_custom_prompt()
    assert isinstance(prompt, PromptTemplate) , "Failed getting the prompt, check test_custom_prompt()"
    return prompt

test_prompt = test_set_custom_prompt()
print(
    test_set_custom_prompt().format(
        context="The sky is blue.",
        query="What is the colour of the sky?"
    )
)


#%% LLMS TESTING #########################################################
def test_load_llm_ctra_llama27b():
    llm = load_llm_ctra_llama27b()
    assert isinstance(llm, CTransformers) , "Failed getting the llm, check test_load_llm_ctra_llama27b()"
    return llm

test_llm = test_load_llm_ctra_llama27b()
#%%
def test_retrieval_qa_chain_from_local_db():
    llm = test_llm
    prompt = set_custom_prompt()
    db = FAISS.load_local("vectorstores/db_faiss", get_generator_embeddings())

    qa_chain = retrieval_qa_chain_from_local_db(llm, prompt, db)
    assert isinstance(qa_chain, RetrievalQA) , "Failed getting the qa_chain, check test_retrieval_qa_chain_from_local_db()"
    return qa_chain


test_qa_chain = test_retrieval_qa_chain_from_local_db()
response = test_qa_chain({"query": QUERY})

# %%

def test_final_result():
    response = final_result(test_qa_chain, QUERY)
    return response

print(final_result("What is the colour of the sky?"))

# %%
