#%%# prepare environement
import os
import time
from huggingface_hub import snapshot_download
from datasets import Dataset

from pathlib import Path

import pandas as pd

from rag_vectorstore import multi_similarity_search_doc
from rag_llms import get_llama2_llm, get_mistral_llm, get_gpt35_llm

from LogSetup import logger
from rag_ragas import retriever_evaluation
from rag_chains import retrieval_qa_chain_from_local_db


from StipVectorStore import StipVectorStore
from StipKnowledgeBase import StipKnowledgeBase, StipHuggingFaceDataset , load_suswiki, load_wikipedia, load_50_qa_dataset
from StipEmbedding import StipEmbedding

from ragas.langchain.evalchain import RagasEvaluatorChain


from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas import evaluate



#### fixed
TEST_QUERY = "what is A/B testing?"
CHUNK_SIZE = 200
CHUNK_OVERLAP_SCALE = 0.1
TOP_K = 3
INDEX_DISTANCE = "l2"
QUESTION_DATASET = load_50_qa_dataset()['train']
FOLDER_PATH ="experiments/Llm/"
VECTORSTORE = StipVectorStore("faiss")
VECTOR_STORE_DATA = VECTORSTORE.load_vectorstore("vectorstores/db_faiss/sustainability-methods-wiki/bge-large-en-v1.5_200_0.1_"+INDEX_DISTANCE)

# model_path = snapshot_download(repo_id="amgadhasan/phi-2",repo_type="model", local_dir="./huggingface_cache/Models/amgadhasan-phi-2", local_dir_use_symlinks=False)


#%% setup LLM
# test load llmnotion
llama2 = get_llama2_llm()
mistral = get_mistral_llm()
gpt35 = get_gpt35_llm()


#%% sanity test, retrieving and generating for 1 question
# test_qa_chain = retrieval_qa_chain_from_local_db(llm=llama2, vectorstore=VECTOR_STORE_DATA) 
# response = test_qa_chain({'query': TEST_QUERY})

# print(test_qa_chain.name)
# print("the query is: ", response['query'])
# print("the result is: ", response['result'])
# print("the source documents are: ", response['source_documents'])


#%% now RETRIEVE for all questions in QUESTION_DATASET
llama2_qa_chain = retrieval_qa_chain_from_local_db(llm=llama2, vectorstore=VECTOR_STORE_DATA)
mistral_qa_chain = retrieval_qa_chain_from_local_db(llm=mistral, vectorstore=VECTOR_STORE_DATA)
gpt35_qa_chain = retrieval_qa_chain_from_local_db(llm=gpt35, vectorstore=VECTOR_STORE_DATA)
pipeline_qa_chain = [llama2_qa_chain, mistral_qa_chain, gpt35_qa_chain]
# pipeline_qa_chain = [llama2_qa_chain]
# QUESTION_DATASET = load_50_qa_dataset()['train']
# QUESTION_DATASET = QUESTION_DATASET[:3]

# %% ## 

def convert_source_documents_to_ragas_contexts(source_documents):
    contexts = []
    for doc in source_documents:
        contexts.append(doc.page_content)
    return contexts


for qa_chain in pipeline_qa_chain:
    output_df = pd.DataFrame()
    # for each question in the dataset
    for i in range(len(QUESTION_DATASET)):
        response = qa_chain({'query' : QUESTION_DATASET['question'][i]})
        response['result'] = response['result'].rstrip('\n') # clean data 
        # contexts_retrieved = convert_source_documents_to_ragas_contexts(response['source_documents'])
        output_df = pd.concat([output_df, 
                               pd.DataFrame([{'query': response['query'],  # in ragas: question
                                              'ground_truths': QUESTION_DATASET['ground_truths'][i] ,  # ground_truth
                                              'result': response['result'], # answer
                                              'source_documents': response['source_documents']}])], 
                                              ignore_index=True) # contexts


    # save output as a csv file and json
    output_df.to_csv(FOLDER_PATH + qa_chain.name + ".csv", index=False) # only for excel
    output_df.to_json(FOLDER_PATH + qa_chain.name + ".json")
    
print("")


#%% clean data #### not needed anymore, but was necessary for the first time
# import pandas as pd
# FOLDER_PATH ="experiments/Llm/"


# # load data from csv
# df = pd.read_csv(FOLDER_PATH + "llama2_FAISS.csv", index_col=False)
# df['result'] = df['result'].str.rstrip('\n')
# df.to_csv(FOLDER_PATH + "llama2_FAISS.csv", index=False)


# df = pd.read_json(FOLDER_PATH + "llama2_FAISS.json")
# df['result'] = df['result'].str.rstrip('\n')
# df.to_json(FOLDER_PATH + "llama2_FAISS.json")


# %% EVALUATE USING RAGAS EVALUATE
faithfulness_chain = RagasEvaluatorChain(metric=faithfulness)
answer_rel_chain = RagasEvaluatorChain(metric=answer_relevancy)
context_rel_chain = RagasEvaluatorChain(metric=context_precision)
context_recall_chain = RagasEvaluatorChain(metric=context_recall)


# load question one-by-one, and then add the ground_truth
# QUESTION_DATASET = QUESTION_DATASET[:3]

test_response = qa_chain({'query' : QUESTION_DATASET['question'][0]})
test_response['ground_truths'] = QUESTION_DATASET['ground_truths'][0]

def evaluate_qa_dataset_with_chain(qa_chain, QUESTION_DATASET):
    output_df = pd.DataFrame()
    for i in range(1, len(QUESTION_DATASET)):
        response = qa_chain({'query' : QUESTION_DATASET['question'][i]})
        response['result'] = response['result'].rstrip('\n') # clean data 
        response['ground_truths'] = QUESTION_DATASET['ground_truths'][i]

        faithfulness_eval = faithfulness_chain(response) # add 'faithfulness_score': 1.0 to the dict
        answer_rel_eval = answer_rel_chain(response) # add answer_relevancy_score': 0.991 to the dict
        context_rel_eval = context_rel_chain(response)# add context_precision_score': 0.8333 to the dict
        context_recall_eval = context_recall_chain(response) # context_recall_score': 1.0 to the dict

        output_df = pd.concat([output_df,
                                pd.DataFrame([{'query': response['query'],  # in ragas: question
                                                'ground_truths': response['ground_truths'] ,  # ground_truth
                                                'result': response['result'], # answer
                                                'source_documents': response['source_documents'],
                                                'faithfulness_score': faithfulness_eval['faithfulness_score'],
                                                'answer_relevancy_score': answer_rel_eval['answer_relevancy_score'],
                                                'context_precision_score': context_rel_eval['context_precision_score'],
                                                'context_recall_score': context_recall_eval['context_recall_score']}])], 
                                                ignore_index=True) # contexts
        
    # save output as a csv file and json
    output_df.to_csv(FOLDER_PATH + qa_chain.name + "_eval.csv", index=False) # only for excel
    output_df.to_json(FOLDER_PATH + qa_chain.name + "_eval.json")

print("")

llama2_eval = evaluate_qa_dataset_with_chain(llama2_qa_chain, QUESTION_DATASET)
mistral_eval = evaluate_qa_dataset_with_chain(mistral_qa_chain, QUESTION_DATASET)
gpt35_eval = evaluate_qa_dataset_with_chain(gpt35_qa_chain, QUESTION_DATASET)



# %%
