# %% ### parameters

import pickle
import pandas as pd


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

#%% trial algorithm using trimmed values  

### trim experiment using pseudo value
# QUESTION_DATASET = QUESTION_DATASET[:10]
# FOLDER_PATH ="experiments/ALL/trim/"
TOP_K = [2]
LLMS = ["llama2", "mistral", "gpt35"]
INDEX_DISTANCES = [eucledian_str, innerproduct_str, cosine_str]
VECTORSTORES = [faiss_str, chroma_str]
EMBEDDINGS = [bge_str, gte_str,]
KNOWLEDGE_BASES = [ suswiki_str, wikipedia_str]

GENERATE_FLAG = False # to generate the answer csv and json - use it mainly for trigerring gpt35
EVALUATE_FLAG = False # to generate ragas evaluation and save it in csv and json


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
                        try:
                            with open(FOLDER_PATH + f"{language_model}_{vector_store_name_experiment_file}_{k}_RagasEval.pkl", 'rb') as f:
                                result = pickle.load(f)

                            new_row = pd.DataFrame({
                                'KB': [knowledge_base],
                                'Embedding': [embedding],
                                'Vector Store': [vector_store_name],
                                'Index Distance': [index_distance],
                                'k Docs': [k],
                                'LLM': [language_model],
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

                        except FileNotFoundError:
                            print(f"File not found: {FOLDER_PATH + f'{language_model}_{vector_store_name_experiment_file}_{k}_RagasEval.pkl'}")

                        




