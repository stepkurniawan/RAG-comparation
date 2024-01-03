"""
Experiment to compare different vector store and see the effect on the retriever performance
"""


#%% 0. Import libraries
# !pip install bs4 chromadb tiktoken faiss-cpu accelerate xformers ragas

import os
import threading
import concurrent.futures

import time

from rag_vectorstore import similarity_search_doc, multi_similarity_search_doc
# from rag_prompting import set_custom_prompt, set_custom_prompt_new, get_formatted_prompt

from rag_splitter import split_data_to_docs
from LogSetup import logger
from rag_ragas import retriever_evaluation

from StipVectorStore import StipVectorStore
from StipKnowledgeBase import StipKnowledgeBase, StipHuggingFaceDataset
from StipEmbedding import StipEmbedding

from ragas import evaluate
from ragas.metrics import ContextPrecision, ContextRecall


import pandas as pd


# 1. UI ##############################################################

# 2. Set up environment ############################################
QUERY = "When and where did the quantitative Content Analysis method originate?"
DB_PATH = "vectorstores/db_faiss"
LINK = "https://sustainabilitymethods.org/index.php/A_matter_of_probability"
OUTPUT_PATH = "experiments/vectorstore_comp"

#%% 3. RETRIEVER #####################################################

# 3.1 Load knowledge base / dataset #######
# data = load_from_webpage("https://sustainabilitymethods.org/index.php/A_matter_of_probability")
suswiki_kb = StipKnowledgeBase("stepkurniawan/sustainability-methods-wiki", None, "huggingface_cache/suswiki_hf")
# wikipedia_kb = KnowledgeBase("wikipedia", "20220301.simple", "huggingface_cache/wikipedia_hf")
dict_data = suswiki_kb.load_documents()
print(f'successful load data from {dict_data["source"]}')

# 3.1.1 limit data for test
# data = data[:10]

# 3.2 Split text into chunks ##############
# docs = split_data_to_docs(data=dict_data, chunk_size=200, chunk_overlap_scale=0.1) # already split in create_vectorsore
docs = dict_data
# 3.3 embedding ###########
embed_model = StipEmbedding("bge").embed_model

#%% 4 VECTOR STORE ######################################################
## create LOCAL FAISS and chroma
def prepare_vectorstore(vectorstore_type: str):
    # 4.1 create vectorstore
    vs = StipVectorStore(vectorstore_type)

    # Create vectorstore
    start_time = time.time()
    vs_data = vs.create_vectorstore(docs, embed_model) 
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed time to create {vectorstore_type}: {elapsed_time} seconds")

    return vs_data, vs

# uncomment this to create vectorstore

def thread_function(vs_type: str):
    try:
        data_vs, vs = prepare_vectorstore(vs_type)
        print(f"finish creating {vs_type} vectorstore")
        return data_vs, vs
    except Exception as e:
        print(f"Exception occurred in thread_function: {e}")

#### threading.Thread() ####
# faiss_thread = threading.Thread(target=thread_function, args=("faiss",))
# chroma_thread = threading.Thread(target=thread_function, args=("chroma",))
# faiss_thread.start()
# chroma_thread.start()
# faiss_thread.join()
# chroma_thread.join()
# faiss_data, faiss_vs = faiss_thread.result
# chroma_data, chroma_vs = chroma_thread.result
# print("finish creating vectorstore")
####################################

# ### concurrent.futures.ThreadPoolExecutor() ####
# with concurrent.futures.ThreadPoolExecutor() as executor:
#     vectorstore_types = ["faiss", "chroma"]
#     results_generator = executor.map(thread_function, vectorstore_types)

# results_list = list(results_generator)
# faiss_future = results_list[0]
# chroma_future = results_list[1]

# try: 
#     faiss_data, faiss_vs = faiss_future[0], faiss_future[1]
#     chroma_data, chroma_vs = chroma_future[0], chroma_future[1]
# except Exception as e:
#     print(f"Exception occurred: {e}")
####################################


########## FIRST TIME ONLE ##########
faiss_data , faiss_vs = prepare_vectorstore("faiss")
chroma_data, chroma_vs = prepare_vectorstore("chroma")

####### LOAD FROM LOCAL ##########
faiss_vs = StipVectorStore("faiss")
faiss_data = faiss_vs.load_vectorstore("vectorstores/db_faiss/sustainability-methods-wiki/bge-large-en-v1.5_200_0.1")
chroma_vs = StipVectorStore("chroma")
chroma_data = chroma_vs.load_vectorstore("vectorstores/db_chroma/sustainability-methods-wiki/bge-large-en-v1.5_200_0.1")



#%% # 4.5 Similiarity search
## uncomment this to do similarity search
database = faiss_data
database_obj = faiss_vs
database_name = "faiss"
database2 = chroma_data
database_obj2 = chroma_vs
database_name2 = "chroma"

similar_docs = similarity_search_doc(database, QUERY, 1)

###################### 4.5 EVALUATE RETRIEVER : context precision , recall, and F-measure

curated_qa_dataset = StipHuggingFaceDataset("stepkurniawan/sustainability-methods-wiki", "50_QA", ["question", "ground_truths"]).load_dataset()
dataset = curated_qa_dataset['train']

# 4.5.1 answer using similarity search
# create a table with question, ground_truths, and context (retrieved_answer)
my_k = 3

def evaluate_retriever(vectorstore_database, qa_dataset, k):
    # 4.5.1 answer using similarity search
    # create a table with question, ground_truths, and context (retrieved_answer)
    start_time = time.time()
    contexted_dataset = multi_similarity_search_doc(vectorstore_database, qa_dataset, k)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"!Note: elapsed time for multi similarity search: {elapsed_time} seconds")

    # answer using ragas evaluate
    context_precision = ContextPrecision()
    context_recall = ContextRecall(batch_size=10)
    contexted_dataset = context_precision.score(contexted_dataset)
    contexted_dataset = context_recall.score(contexted_dataset)

    contexted_df = pd.DataFrame(contexted_dataset)
    return contexted_df

def save_locally(contexted_df, database_obj):
    # Check if the directory exists, if not, create it
    file_path = f"{OUTPUT_PATH}/{database_obj.vectorstore_name}_{database_obj.docs_source}_{database_obj.model_name}_{database_obj.chunk_size}_{database_obj.chunk_overlap_scale}_k{my_k}"
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    contexted_df.to_csv(file_path +".csv", sep="|", )
    #save answer to json
    contexted_df.to_json(file_path+".json")



contexted_df = evaluate_retriever(database, dataset, my_k)
contexted_df2 = evaluate_retriever(database2, dataset, my_k)
save_locally(contexted_df, database_obj)
save_locally(contexted_df2, database_obj2)
    
# %%
import pandas as pd 

def read_local_results(docs_source, model_name, chunk_size, chunk_overlap_scale, k):
    output_df = pd.DataFrame() # this will countain columns: question, ground_truths, contexts, context_precision_FAISS, context_recall_FAISS, context_precision_Chroma, context_recall_Chroma
    faiss_df = pd.read_csv(f"{OUTPUT_PATH}/faiss_{docs_source}_{model_name}_{chunk_size}_{chunk_overlap_scale}_k{k}.csv", sep="|", )
    chroma_df = pd.read_csv(f"{OUTPUT_PATH}/chroma_{docs_source}_{model_name}_{chunk_size}_{chunk_overlap_scale}_k{k}.csv", sep="|", )

    vectorstores = ["faiss", "chroma"]
    # get the columns: question, ground_truths, contexts, context_precision_FAISS, context_recall_FAISS from each vectorstore
    for vectorstore in vectorstores:
        # load from local
        file_path = f"{OUTPUT_PATH}/{vectorstore}_{docs_source}_{model_name}_{chunk_size}_{chunk_overlap_scale}_k{k}"
        print("load from :", file_path)
        df = pd.read_csv(file_path +".csv", sep="|", )
        df = df.drop(columns=['Unnamed: 0'])
        df = df.rename(columns={"contexts": f"contexts_{vectorstore}",
                                "context_precision": f"context_precision_{vectorstore}", 
                                "context_recall": f"context_recall_{vectorstore}"})
        output_df = pd.concat([output_df, df], axis=1)

    return output_df, faiss_df, chroma_df
        

output_df, faiss_df, chroma_df = read_local_results("sustainability-methods-wiki", "bge-large-en-v1.5", "200", "0.1", my_k)
print(output_df)


# %% # 
faiss_mean_context_precision = faiss_df['context_precision'].mean()
faiss_mean_context_recall = faiss_df['context_recall'].mean()
faiss_f_measure = 2 * faiss_mean_context_precision * faiss_mean_context_recall / (faiss_mean_context_precision + faiss_mean_context_recall)
chroma_mean_context_precision = chroma_df['context_precision'].mean()
chroma_mean_context_recall = chroma_df['context_recall'].mean()
chroma_f_measure = 2 * chroma_mean_context_precision * chroma_mean_context_recall / (chroma_mean_context_precision + chroma_mean_context_recall)

# create a table to store all those value above
final_df = pd.DataFrame({"vectorstore": ["faiss", "chroma"],
                            "context_precision": [faiss_mean_context_precision, chroma_mean_context_precision],
                            "context_recall": [faiss_mean_context_recall, chroma_mean_context_recall],
                            "f_measure": [faiss_f_measure, chroma_f_measure]})
print(final_df)


# %%
