#%%# prepare environement
import os
import time

import pandas as pd
import pickle
from datasets import Dataset, Value, Sequence

from rag_vectorstore import multi_similarity_search_doc
from rag_llms import get_llama2_llm, get_mistral_llm, get_gpt35_llm
from rag_chains import generate_context_answer_langchain, retrieval_qa_chain_from_local_db
from rag_ragas import evaluate_qa_dataset_with_response


from LogSetup import logger
from rag_ragas import retriever_evaluation

from StipVectorStore import StipVectorStore
from StipKnowledgeBase import StipKnowledgeBase, StipHuggingFaceDataset , load_suswiki, load_wikipedia, load_50_qa_dataset
from StipEmbedding import StipEmbedding

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness
)

#### fixed
CHUNK_SIZE = 200 # since we know the lower it is, the better
CHUNK_OVERLAP_SCALE = 0.1
TOP_K = [2,3,4] # these are 3 best topk
QUESTION_DATASET = load_50_qa_dataset()['train']
FOLDER_PATH ="experiments/ALL/"

llama2 = get_llama2_llm()
mistral = get_mistral_llm()
gpt35 = get_gpt35_llm()
LLMS = [llama2, mistral, gpt35]

faiss_vs = StipVectorStore("faiss")
chroma_vs = StipVectorStore("chroma")

suswiki_kb = load_suswiki()
wikipedia_kb = load_wikipedia()




# %% ### parameters

suswiki_str = "sustainability-methods-wiki"
wikipedia_str = "wikipedia"
KNOWLEDGE_BASES = [suswiki_str, wikipedia_str]

bge_str = "bge-large-en-v1.5"
gte_str = "gte-large"
uae_str = "UAE-Large-V1"
EMBEDDINGS = [bge_str, gte_str, uae_str]

faiss_str = "db_faiss"
chroma_str = "db_chroma"
VECTORSTORES = [faiss_str, chroma_str]

eucledian_str = "l2"
cosine_str = "cosine"
innerproduct_str = "ip"
INDEX_DISTANCES = [eucledian_str, cosine_str, innerproduct_str]




# %% CHECK and PREPARE VECTOR STORE
# for kb in KNOWLEDGE_BASES:
#     for emb in EMBEDDINGS:
#         for vs in VECTORSTORES:
#             for dist in INDEX_DISTANCES:
#                 if vs == faiss_str:
#                     vectorstore = faiss_vs
#                 elif vs == chroma_str:
#                     vectorstore = chroma_vs

#                 vectorstore_path = f"vectorstores/{vs}/{kb}/{emb}_{CHUNK_SIZE}_{CHUNK_OVERLAP_SCALE}_{dist}"
#                 print(f"Loading vectorstore {vectorstore_path}")

#                 # check if path exists
#                 if os.path.exists(vectorstore_path):
#                     vectorstore_data = vectorstore.load_vectorstore(vectorstore_path)
#                 else:
#                     print(f"WARNING!   Vectorstore {vectorstore_path} does not exist")
#                     print(f"Creating vector store {vectorstore_path}...")
#                     kb_data = suswiki_kb if kb == suswiki_str else wikipedia_kb
#                     embedding_code = emb.split("-")[0].lower()
#                     emb_model = StipEmbedding(embedding_code).embed_model
#                     vectorstore_data = vectorstore.create_vectorstore(kb_data, emb_model, CHUNK_SIZE, CHUNK_OVERLAP_SCALE, dist)

#                     continue


# faiss_data = faiss_vs.load_vectorstore("vectorstores/db_faiss/sustainability-methods-wiki/bge-large-en-v1.5_200_0.1_"+INDEX_DISTANCE)

#%% trial algorithm using trimmed values  

### trim experiment using pseudo value
# QUESTION_DATASET = QUESTION_DATASET[:10]
# FOLDER_PATH ="experiments/ALL/trim/"
TOP_K = [2]
LLMS = [mistral, llama2]
VECTORSTORES = [faiss_str]
KNOWLEDGE_BASES = [ suswiki_str, wikipedia_str]
EMBEDDINGS = [gte_str,]
INDEX_DISTANCES = [eucledian_str]

GENERATE_FLAG = False # to generate the answer csv and json - use it mainly for trigerring gpt35
EVALUATE_FLAG = True # to generate ragas evaluation and save it in csv and json

def load_or_create_vectorstore(vector_store_name, vector_store_path, knowledge_base, embedding, index_distance):
    if vector_store_name == faiss_str:
        current_vector_store = faiss_vs
    elif vector_store_name == chroma_str:
        current_vector_store = chroma_vs
    
    if os.path.exists(vector_store_path):
        return current_vector_store.load_vectorstore(vector_store_path)
    else:
        print(f"WARNING! Vectorstore {vector_store_path} does not exist")
        print(f"WARNING! Creating vector store {vector_store_path}...")
        
        kb_data = suswiki_kb if knowledge_base == suswiki_str else wikipedia_kb
        embedding_code = embedding.split("-")[0].lower()
        emb_model = StipEmbedding(embedding_code).embed_model

        return current_vector_store.create_vectorstore(kb_data, emb_model, CHUNK_SIZE, CHUNK_OVERLAP_SCALE, index_distance)

def run_all(KNOWLEDGE_BASES, EMBEDDINGS, VECTORSTORES, INDEX_DISTANCES, TOP_K, LLMS, QUESTION_DATASET, FOLDER_PATH):
    for knowledge_base in KNOWLEDGE_BASES:
        for embedding in EMBEDDINGS:
            for vector_store_name in VECTORSTORES:
                for index_distance in INDEX_DISTANCES:
                    
                    # Construct the path to the vector store
                    vector_store_path = f"vectorstores/{vector_store_name}/{knowledge_base}/{embedding}_{CHUNK_SIZE}_{CHUNK_OVERLAP_SCALE}_{index_distance}"
                    vector_store_data = load_or_create_vectorstore(vector_store_name, vector_store_path, knowledge_base, embedding, index_distance).db
                    
                    # Iterate over all values of k in TOP_K
                    for k in TOP_K:
                        # Iterate over all language models in LLMS
                        for language_model in LLMS:
                            # Print a message indicating that retrieval is being performed
                            print(f"Retrieving for {language_model.name} using {vector_store_path}...")
                            logger.info(f"Retrieving for {language_model.name} using {vector_store_path}...")
                            

                            # # Generate answer result from QA dataset
                            FOLDER_PATH = f"experiments/ALL/{knowledge_base}/{embedding}/{vector_store_name}/{index_distance}/"
                            if FOLDER_PATH is not None:
                                os.makedirs(FOLDER_PATH, exist_ok=True)
                                
                            generate_df = generate_context_answer_langchain(QUESTION_DATASET, language_model, vector_store_data, k, folder_save_path=FOLDER_PATH) # create csv and json with name: folder_save_path + qa_chain.name + "_" + str(k) + "_gen.json"
                            
                            # Evaluate the QA dataset with the QA chain and create an output dataframe
                            # output_df = evaluate_qa_dataset_with_response(generate_df, QUESTION_DATASET, FOLDER_PATH)
                            
                            # Print a message indicating that the output has been created
                            print("output created in path:", FOLDER_PATH)
                            logger.info(f"output created in path: {FOLDER_PATH}, check for CSV and JSON {language_model.name} in {vector_store_path} ")


# uncomment run_all to create a csv and json file. Right now, the ones that are missing are from GPT35
if GENERATE_FLAG:
    run_all(KNOWLEDGE_BASES, EMBEDDINGS, VECTORSTORES, INDEX_DISTANCES, TOP_K, LLMS, QUESTION_DATASET, FOLDER_PATH)

# %% Optional: FIX some answer


#%% RAGAS Evaluation JSON dataframe
def evaluate_json_from_generated_contexts(file_name):
    """
    input: json file name
    output: ragas evaluation result, and save it as csv and json
    """
    #### Create dataset from json file
    # create a dataframe from json file
    df = pd.read_json(file_name)

    # change column names query -> question, ground_truths -> ground_truth, result -> answer, source_documents -> contexts
    df = df.rename(columns={'query': 'question', 
                            'ground_truths': 'ground_truth', 
                            'result': 'answer', 
                            'source_documents': 'contexts'})
    
    # clean context into just page_content
    for i in range(len(df)):
        for j in range(len(df['contexts'][i])):
            if isinstance(df['contexts'][i][j], dict):
                df['contexts'][i][j] = df['contexts'][i][j]['page_content']
    
    data_dict = {
        'question': df['question'].tolist(),
        'ground_truth': df['ground_truth'].tolist(),
        'contexts': df['contexts'].tolist(),
        'answer': df['answer'].tolist(),
    }

    dataset = Dataset.from_dict(data_dict)

    # Evaluate dataset
    result = evaluate(
        dataset,
        metrics = [
            answer_relevancy,
            faithfulness,
            context_recall,
            context_precision,
            answer_correctness
            ]
    )

    # save result object as pickle dump
    with open(FOLDER_PATH + f"{language_model.name}_{vector_store_name_experiment_file}_{k}_RagasEval.pkl", 'wb') as f:
        pickle.dump(result, f)

    # create dataframe and save it to csv and json file
    result_df = result.to_pandas()
    result_df.to_csv(FOLDER_PATH + f"{language_model.name}_{vector_store_name_experiment_file}_{k}_RagasEval.csv")
    result_df.to_json(FOLDER_PATH + f"{language_model.name}_{vector_store_name_experiment_file}_{k}_RagasEval.json")
    print(f"evaluation result saved in {FOLDER_PATH + language_model.name}_{vector_store_name_experiment_file}_{k}_eval.csv")

    return result

if EVALUATE_FLAG:
    for knowledge_base in KNOWLEDGE_BASES:
            for embedding in EMBEDDINGS:
                for vector_store_name in VECTORSTORES:
                    for index_distance in INDEX_DISTANCES:
                        # Iterate over all values of k in TOP_K
                        for k in TOP_K:
                            # Iterate over all language models in LLMS
                            for language_model in LLMS:
                                FOLDER_PATH = f"experiments/ALL/{knowledge_base}/{embedding}/{vector_store_name}/{index_distance}/"
                                if vector_store_name == faiss_str:
                                    vector_store_name_experiment_file = "FAISS"
                                elif vector_store_name == chroma_str:
                                    vector_store_name_experiment_file = "Chroma"

                                file_name = FOLDER_PATH + f"{language_model.name}_{vector_store_name_experiment_file}_{k}_gen.json"

                                ### evaluate json 
                                
                                evaluate_json_from_generated_contexts(file_name)
                                




#%% presents the results

# create a result dataframe
all_result_df = pd.DataFrame()

for knowledge_base in KNOWLEDGE_BASES:
    for embedding in EMBEDDINGS:
        for vector_store_name in VECTORSTORES:
            for index_distance in INDEX_DISTANCES:
                for k in TOP_K:
                    for language_model in LLMS:
                        FOLDER_PATH = f"experiments/ALL/{knowledge_base}/{embedding}/{vector_store_name}/{index_distance}/"
                        if vector_store_name == faiss_str:
                            vector_store_name_experiment_file = "FAISS"
                        elif vector_store_name == chroma_str:
                            vector_store_name_experiment_file = "Chroma"

                        # load evaluation result
                        with open(FOLDER_PATH + f"{language_model.name}_{vector_store_name_experiment_file}_{k}_RagasEval.pkl", 'rb') as f:
                            result = pickle.load(f)

                        new_row = pd.DataFrame({
                            'KB': [knowledge_base],
                            'Embedding': [embedding],
                            'Vector Store': [vector_store_name],
                            'Index Distance': [index_distance],
                            'k Docs': [k],
                            'LLM': [language_model.name],
                            'Answer Relevancy': [round(result['answer_relevancy'], 3)],
                            'Faithfulness': [round(result['faithfulness'], 3)],
                            'Context Recall': [round(result['context_recall'], 3)],
                            'Context Precision': [round(result['context_precision'], 3)],
                            'Answer Correctness': [round(result['answer_correctness'], 3)]
                        }) # input of the DataFrame is a dictionary

                        # Define the mappings
                        mappings = {
                            'KB': {
                                'sustainability-methods-wiki': 'Suswiki',
                                'wikipedia': 'Wikipedia'
                            },
                            'Vector Store': {
                                'db_faiss': 'FAISS',
                                'db_chroma': 'Chroma'
                            },
                            'Embedding': {
                                'bge-large-en-v1.5': 'BGE',
                                'gte-large': 'GTE',
                                'UAE-Large-V1': 'UAE'
                            },
                            'Index Distance': {
                                'l2': 'Eucledian',
                                'cosine': 'Cosine',
                                'ip': 'Inner Product'
                            },
                            'LLM': {
                                'gpt35': 'GPT-3.5',
                                'mistral': 'Mistral',
                                'llama2': 'Llama2'
                            }
                        }

                        # Apply the mappings
                        for column, mapping in mappings.items():
                            new_row[column] = new_row[column].map(mapping).fillna(new_row[column])

                        all_result_df = pd.concat([all_result_df, new_row])
                        all_result_df = all_result_df.reset_index(drop=True)

                        






                    


#%% testing using GPU
import torch
import torch.nn as nn

# Define your model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        output = self.fc(x)
        return output

model = MyModel()


if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # Use torch.nn.DataParallel to wrap your model
    model = nn.DataParallel(model)
# %%
