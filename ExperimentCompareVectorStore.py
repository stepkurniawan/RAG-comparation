"""
Experiment to compare different vector store and see the effect on the retriever performance
"""


#%% 0. Import libraries
# !pip install bs4 chromadb tiktoken faiss-cpu accelerate xformers ragas

import os
import threading
import concurrent.futures
import matplotlib.pyplot as plt


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
QUERY = "What is the ANOVA powerful for?"
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
# faiss_data , faiss_vs = prepare_vectorstore("faiss")
# chroma_data, chroma_vs = prepare_vectorstore("chroma")

####### LOAD FROM LOCAL ##########
faiss_vs = StipVectorStore("faiss")
faiss_vs.load_vectorstore("vectorstores/db_faiss/sustainability-methods-wiki/bge-large-en-v1.5_200_0.1")
faiss_data = faiss_vs.db
chroma_vs = StipVectorStore("chroma")
chroma_vs.load_vectorstore("vectorstores/db_chroma/sustainability-methods-wiki/bge-large-en-v1.5_200_0.1")
chroma_data = chroma_vs.db


#%% # 4.5 Similiarity search
## uncomment this to do similarity search
database = faiss_data
database_obj = faiss_vs
database_name = "faiss"
database2 = chroma_data
database_obj2 = chroma_vs
database_name2 = "chroma"

similar_docs = similarity_search_doc(database2, QUERY, 9)

###################### 4.5 EVALUATE RETRIEVER : context precision , recall, and F-measure

curated_qa_dataset = StipHuggingFaceDataset("stepkurniawan/sustainability-methods-wiki", "50_QA_reviewed", ["question", "ground_truths"]).load_dataset()
dataset = curated_qa_dataset['train']

# 4.5.1 answer using similarity search
# create a table with question, ground_truths, and context (retrieved_answer)


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
    file_path = f"{OUTPUT_PATH}/{database_obj.vectorstore_name}_{database_obj.docs_source}_{database_obj.embedding_name}_{database_obj.chunk_size}_{database_obj.chunk_overlap_scale}_k{my_k}"
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    contexted_df.to_csv(file_path +".csv", sep="|", )
    #save answer to json
    contexted_df.to_json(file_path+".json")

#### Uncomment this to evaluate retriever (use OpenAI API)
my_k = 1
# faiss_contexted_result_df = evaluate_retriever(database, dataset, my_k)
# chroma_contexted_result_df = evaluate_retriever(database2, dataset, my_k)
# save_locally(faiss_contexted_result_df, faiss_vs)
# save_locally(chroma_contexted_result_df, chroma_vs)


#### Uncomment this to evaluate retriever (loop)
# # # do a for loop for k from 1 - 10
# for my_k in range(1, 6):
#     print(f"evaluate for k: {my_k}--------------------")

#     faiss_contexted_result_df = evaluate_retriever(database, dataset, my_k)
#     chroma_contexted_result_df = evaluate_retriever(database2, dataset, my_k)
#     save_locally(faiss_contexted_result_df, faiss_vs)
#     save_locally(chroma_contexted_result_df, chroma_vs)
    
# %%

def read_local_results(docs_source, embedding_name, chunk_size, chunk_overlap_scale, k):
    output_df = pd.DataFrame() # this will countain columns: question, ground_truths, contexts, context_precision_FAISS, context_recall_FAISS, context_precision_Chroma, context_recall_Chroma
    faiss_df = pd.read_csv(f"{OUTPUT_PATH}/faiss_{docs_source}_{embedding_name}_{chunk_size}_{chunk_overlap_scale}_k{k}.csv", sep="|", )
    chroma_df = pd.read_csv(f"{OUTPUT_PATH}/chroma_{docs_source}_{embedding_name}_{chunk_size}_{chunk_overlap_scale}_k{k}.csv", sep="|", )

    vectorstores = ["faiss", "chroma"]
    # get the columns: question, ground_truths, contexts, context_precision_FAISS, context_recall_FAISS from each vectorstore
    for vectorstore in vectorstores:
        # load from local
        file_path = f"{OUTPUT_PATH}/{vectorstore}_{docs_source}_{embedding_name}_{chunk_size}_{chunk_overlap_scale}_k{k}"
        print("load from :", file_path)
        df = pd.read_csv(file_path +".csv", sep="|", )
        df = df.drop(columns=['Unnamed: 0'])
        df = df.rename(columns={"contexts": f"contexts_{vectorstore}",
                                "context_precision": f"context_precision_{vectorstore}", 
                                "context_recall": f"context_recall_{vectorstore}"})
        output_df = pd.concat([output_df, df], axis=1)

    return output_df, faiss_df, chroma_df
        

output_df, faiss_df, chroma_df = read_local_results("sustainability-methods-wiki", "bge-large-en-v1.5", "200", "0.1", my_k)
# print(output_df)


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
print("current_k = ", my_k)
print(final_df)


# %% LOAD 7 results from local

# loop for k from 1 - 7
final_df = pd.DataFrame()
for k in range(1, 8):
    print(f"evaluate for k: {k}--------------------")

    output_df, faiss_df, chroma_df = read_local_results("sustainability-methods-wiki", "bge-large-en-v1.5", "200", "0.1", k)
    faiss_mean_context_precision = faiss_df['context_precision'].mean()
    faiss_mean_context_recall = faiss_df['context_recall'].mean()
    faiss_f_measure = 2 * faiss_mean_context_precision * faiss_mean_context_recall / (faiss_mean_context_precision + faiss_mean_context_recall)
    chroma_mean_context_precision = chroma_df['context_precision'].mean()
    chroma_mean_context_recall = chroma_df['context_recall'].mean()
    chroma_f_measure = 2 * chroma_mean_context_precision * chroma_mean_context_recall / (chroma_mean_context_precision + chroma_mean_context_recall)

    # concat the result to final_df
    df = pd.DataFrame({ "k": [k,k],
                        "vectorstore": ["faiss", "chroma"],
                        "context_precision": [faiss_mean_context_precision, chroma_mean_context_precision],
                        "context_recall": [faiss_mean_context_recall, chroma_mean_context_recall],
                        "f_measure": [faiss_f_measure, chroma_f_measure]})
    final_df = pd.concat([final_df, df], axis=0)
    
    print("current_k = ", k)

    print(final_df)


# Filter the DataFrame for each vectorstore
faiss_df = final_df[final_df['vectorstore'] == 'faiss']
chroma_df = final_df[final_df['vectorstore'] == 'chroma']



#%% CREATE VISUALIZATION

reddishbrown = '#8B4513'
# FAISS variance error bar for context precision and context recall
# plt.errorbar(faiss_df['k'], faiss_df['context_precision'], yerr=faiss_df['context_precision'].std(), fmt='-', color='skyblue', label='FAISS context_precision with stddev')
# plt.errorbar(chroma_df['k'], chroma_df['context_precision'], yerr=chroma_df['context_precision'].std(), fmt='-', color=reddishbrown, label='Chroma context_precision with stddev')

# plt.errorbar(faiss_df['k'], faiss_df['context_recall'], yerr=faiss_df['context_recall'].std(), fmt='--', color='skyblue', label='FAISS context_recall with stddev')
# plt.errorbar(chroma_df['k'], chroma_df['context_recall'], yerr=chroma_df['context_recall'].std(), fmt='--', color=reddishbrown, label='Chroma context_recall with stddev')

# Add labels and title
plt.title("Context Precision and Recall for different Vector Store based on k")
plt.xlabel("k")
plt.ylabel("context_precision and context_recall score")



#### Add labels at each point
# Plot the data and error bars
plt.errorbar(faiss_df['k'], faiss_df['context_precision'], yerr=faiss_df['context_precision'].std(), fmt='-', color='skyblue')
for i in range(len(faiss_df)):
    plt.annotate(f"{faiss_df['context_precision'].values[i]:.2f}", (faiss_df['k'].values[i], faiss_df['context_precision'].values[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.errorbar(faiss_df['k'], faiss_df['context_recall'], yerr=faiss_df['context_recall'].std(), fmt='--', color='skyblue')
for i in range(len(faiss_df)):
    plt.annotate(f"{faiss_df['context_recall'].values[i]:.2f}", (faiss_df['k'].values[i], faiss_df['context_recall'].values[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.errorbar(chroma_df['k'], chroma_df['context_precision'], yerr=chroma_df['context_precision'].std(), fmt='-', color=reddishbrown)
for i in range(len(chroma_df)):
    plt.annotate(f"{chroma_df['context_precision'].values[i]:.2f}", (chroma_df['k'].values[i], chroma_df['context_precision'].values[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.errorbar(chroma_df['k'], chroma_df['context_recall'], yerr=chroma_df['context_recall'].std(), fmt='--', color=reddishbrown)
for i in range(len(chroma_df)):
    plt.annotate(f"{chroma_df['context_recall'].values[i]:.2f}", (chroma_df['k'].values[i], chroma_df['context_recall'].values[i]), textcoords="offset points", xytext=(0,10), ha='center')
#### Create custom legend

import matplotlib.lines as mlines

orange_line = mlines.Line2D([], [], color='skyblue', label='FAISS')
reddishbrown_line = mlines.Line2D([], [], color=reddishbrown, label='Chroma')
solid_line = mlines.Line2D([], [], color='black', linestyle='-', label='context_precision')
dotted_line = mlines.Line2D([], [], color='black', linestyle='--', label='context_recall')
plt.legend(handles=[orange_line, reddishbrown_line, solid_line, dotted_line], bbox_to_anchor=(1.05, 1), loc='upper left')


# Show the plot
plt.show()

# %%
