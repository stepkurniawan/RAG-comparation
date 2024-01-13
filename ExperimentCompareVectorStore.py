"""
Experiment to compare different vector store and see the effect on the retriever performance

Structure:
1. Create vectorstore
2. Load vectorstore (check path)
3. Evaluate retriever
4. Visualize the result

"""


#%% 0. Import libraries
# !pip install bs4 chromadb tiktoken faiss-cpu accelerate xformers ragas

import os
from dotenv import load_dotenv
load_dotenv()

import threading
import concurrent.futures
import matplotlib.pyplot as plt
from adjustText import adjust_text # better annotation
import numpy as np


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
import ast



# 2. Set up environment ############################################
QUERY = "What is the advantage of A/B testing?"
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
    vs_data = vs.create_vectorstore(docs, embed_model,200,0.1, index_distance="l2") 
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed time to create {vectorstore_type}: {elapsed_time} seconds")

    return vs_data, vs


########## FIRST TIME ONLE ##########
#### Uncomment this if needed ######
# faiss_data , faiss_vs = prepare_vectorstore("faiss")
# chroma_data, chroma_vs = prepare_vectorstore("chroma")

####### LOAD FROM LOCAL ##########
faiss_vs = StipVectorStore("faiss")
faiss_vs.load_vectorstore("vectorstores/db_faiss/sustainability-methods-wiki/bge-large-en-v1.5_200_0.1_l2")
faiss_data = faiss_vs.db

chroma_vs = StipVectorStore("chroma")
chroma_vs.load_vectorstore("vectorstores/db_chroma/sustainability-methods-wiki/bge-large-en-v1.5_200_0.1_l2")
chroma_data = chroma_vs.db


#%% # SIMILARITY SEARCH ##############################################



similar_docs = similarity_search_doc(chroma_data, QUERY, 6)

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
my_k = 6
# faiss_contexted_result_df = evaluate_retriever(faiss_data, dataset, my_k)
chroma_contexted_result_df = evaluate_retriever(chroma_data, dataset, my_k)
# save_locally(faiss_contexted_result_df, faiss_vs)
# save_locally(chroma_contexted_result_df, chroma_vs)


#### Uncomment this to evaluate retriever (loop)
# # do a for loop for k from 1 - 10
# for my_k in range(4, 8):
#     print(f"evaluate for k: {my_k}--------------------")

#     faiss_contexted_result_df = evaluate_retriever(faiss_data, dataset, my_k)
#     chroma_contexted_result_df = evaluate_retriever(chroma_data, dataset, my_k)
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
max_k = 6
for k in range(1, max_k):
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
plt.xlabel("number of top document k")
plt.ylabel("context_precision and context_recall score")



#### Add labels at each point
texts = []

# Plot the data and error bars

plt.errorbar(faiss_df['k'], faiss_df['context_precision'], yerr=faiss_df['context_precision'].std(), fmt='-', color='skyblue')
for i in range(len(faiss_df)):
    texts.append(plt.text(faiss_df['k'].values[i], faiss_df['context_precision'].values[i], f"{faiss_df['context_precision'].values[i]:.2f}", ha='center'))

plt.errorbar(faiss_df['k'], faiss_df['context_recall'], yerr=faiss_df['context_recall'].std(), fmt='--', color='skyblue')
for i in range(len(faiss_df)):
    texts.append(plt.text(faiss_df['k'].values[i], faiss_df['context_recall'].values[i], f"{faiss_df['context_recall'].values[i]:.2f}", ha='center'))

plt.errorbar(chroma_df['k'], chroma_df['context_precision'], yerr=chroma_df['context_precision'].std(), fmt='-', color=reddishbrown)
for i in range(len(chroma_df)):
    texts.append(plt.text(chroma_df['k'].values[i], chroma_df['context_precision'].values[i], f"{chroma_df['context_precision'].values[i]:.2f}", ha='center'))

plt.errorbar(chroma_df['k'], chroma_df['context_recall'], yerr=chroma_df['context_recall'].std(), fmt='--', color=reddishbrown)
for i in range(len(chroma_df)):
    texts.append(plt.text(chroma_df['k'].values[i], chroma_df['context_recall'].values[i], f"{chroma_df['context_recall'].values[i]:.2f}", ha='center'))



# # just a line plot instead of error bar
# plt.plot(faiss_df['k'], faiss_df['context_precision'], '-', color='skyblue')
# for i in range(len(faiss_df)):
#     texts.append(plt.text(faiss_df['k'].values[i], faiss_df['context_precision'].values[i], f"{faiss_df['context_precision'].values[i]:.2f}", ha='center'))

# plt.plot(faiss_df['k'], faiss_df['context_recall'], '--', color='skyblue')
# for i in range(len(faiss_df)):
#     texts.append(plt.text(faiss_df['k'].values[i], faiss_df['context_recall'].values[i], f"{faiss_df['context_recall'].values[i]:.2f}", ha='center'))

# plt.plot(chroma_df['k'], chroma_df['context_precision'], '-', color=reddishbrown)
# for i in range(len(chroma_df)):
#     texts.append(plt.text(chroma_df['k'].values[i], chroma_df['context_precision'].values[i], f"{chroma_df['context_precision'].values[i]:.2f}", ha='center'))

# plt.plot(chroma_df['k'], chroma_df['context_recall'], '--', color=reddishbrown)
# for i in range(len(chroma_df)):
#     texts.append(plt.text(chroma_df['k'].values[i], chroma_df['context_recall'].values[i], f"{chroma_df['context_recall'].values[i]:.2f}", ha='center'))




#### Create custom legend
import matplotlib.lines as mlines

orange_line = mlines.Line2D([], [], color='skyblue', label='FAISS')
reddishbrown_line = mlines.Line2D([], [], color=reddishbrown, label='Chroma')
solid_line = mlines.Line2D([], [], color='black', linestyle='-', label='context_precision')
dotted_line = mlines.Line2D([], [], color='black', linestyle='--', label='context_recall')
plt.legend(handles=[orange_line, reddishbrown_line, solid_line, dotted_line], bbox_to_anchor=(1.05, 1), loc='upper left')


#### change the x tick
# Get the current x-axis limits
xmin, xmax = plt.xlim()
# Create a range of integer values from xmin to xmax
xticks = np.arange(np.ceil(xmin), np.floor(xmax)+1)
# Set the new x-axis ticks
plt.xticks(xticks)

# Show the plot
plt.show()

# %% MANUAL EXPERIMENT
import ast

question_row=0
k = 6
# load document from local based on k
output_df_man, faiss_df_man, chroma_df_man = read_local_results("sustainability-methods-wiki", "bge-large-en-v1.5", "200", "0.1", k)

# grab the first row
current_row = output_df_man.iloc[question_row]

faiss_context = current_row['contexts_faiss'] # string
faiss_context_list = ast.literal_eval(faiss_context)

chroma_context = current_row['contexts_chroma']
chroma_context_list = ast.literal_eval(chroma_context)




print("")
# %%
