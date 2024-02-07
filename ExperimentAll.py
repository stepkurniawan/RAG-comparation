#%%# prepare environement
import os
import time

import pandas as pd

from rag_vectorstore import multi_similarity_search_doc
from rag_llms import get_llama2_llm, get_mistral_llm, get_gpt35_llm
from rag_chains import generate_context_answer_langchain, retrieval_qa_chain_from_local_db
from rag_ragas import evaluate_qa_dataset_with_response


from LogSetup import logger
from rag_ragas import retriever_evaluation

from StipVectorStore import StipVectorStore
from StipKnowledgeBase import StipKnowledgeBase, StipHuggingFaceDataset , load_suswiki, load_wikipedia, load_50_qa_dataset
from StipEmbedding import StipEmbedding

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    AnswerCorrectness
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




# %% ### load vectorstore locally

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
#             for dist in INDEX_DISTANCE:
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

#%% try the algorithm using trimmed values

### trim experiment using pseudo value
# QUESTION_DATASET = QUESTION_DATASET[:2]
# FOLDER_PATH ="experiments/ALL/trim/"
# TOP_K = [2,3,4]
LLMS = [llama2,mistral]
# VECTORSTORES = [chroma_str]
# KNOWLEDGE_BASES = [suswiki_str]
# EMBEDDINGS = [bge_str,gte_str]
# INDEX_DISTANCES = [eucledian_str, cosine_str]


for knowledge_base in KNOWLEDGE_BASES:
    for embedding in EMBEDDINGS:
        for vector_store_name in VECTORSTORES:
            for index_distance in INDEX_DISTANCES:
                # Load the appropriate vector store based on the current vector store name
                # I need this, because I want to use the string as the path instead of using the vector store object
                if vector_store_name == faiss_str:
                    current_vector_store = faiss_vs
                elif vector_store_name == chroma_str:
                    current_vector_store = chroma_vs
                
                # Construct the path to the vector store
                vector_store_path = f"vectorstores/{vector_store_name}/{knowledge_base}/{embedding}_{CHUNK_SIZE}_{CHUNK_OVERLAP_SCALE}_{index_distance}"
                
                # Print a message indicating that the vector store is being loaded
                print(f"Loading vectorstore {vector_store_path}...")
                
                # Check if the vector store exists at the constructed path
                if os.path.exists(vector_store_path):
                    #### Important: Load the vector store data
                    vector_store_data = current_vector_store.load_vectorstore(vector_store_path)
                else:
                    # If it doesn't exist, create it
                    print(f"WARNING! Vectorstore {vector_store_path} does not exist")
                    print(f"WARNING! Creating vector store {vector_store_path}...")
                    kb_data = suswiki_kb if knowledge_base == suswiki_str else wikipedia_kb
                    embedding_code = embedding.split("-")[0].lower()
                    emb_model = StipEmbedding(embedding_code).embed_model

                    vectorstore_data = current_vector_store.create_vectorstore(kb_data, emb_model, CHUNK_SIZE, CHUNK_OVERLAP_SCALE, index_distance)
                    logger.info(f"!! Just created Vectorstore: {vector_store_path}")

                # Iterate over all values of k in TOP_K
                for k in TOP_K:
                    # Iterate over all language models in LLMS
                    for language_model in LLMS:
                        # Print a message indicating that retrieval is being performed
                        print(f"Retrieving for {language_model.name} using {vector_store_path}...")
                        logger.info(f"Retrieving for {language_model.name} using {vector_store_path}...")
                        

                        # # Generate answer result from QA dataset
                        generate_df = generate_context_answer_langchain(QUESTION_DATASET, language_model, vector_store_data, k, save_path=FOLDER_PATH)
                        
                        # Evaluate the QA dataset with the QA chain and create an output dataframe
                        # output_df = evaluate_qa_dataset_with_response(generate_df, QUESTION_DATASET, FOLDER_PATH)
                        
                        # Print a message indicating that the output has been created
                        print("output created in path:", FOLDER_PATH)
                        logger.info(f"output created in path: {FOLDER_PATH}, check for CSV and JSON {language_model.name} in {vector_store_path} ")


                

#%%
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
