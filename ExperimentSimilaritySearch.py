"""
Experiment to compare different Similarity Search Algorithm and see the effect on the retriever performance
The similarity search algorithm are:
1. Eucledian Distance
2. Cosine Similarity
3. SVM
"""
# compare with 1 knowledge base first, and then compare with 2 knowledge bases, 
# if there is no significant difference, then we can use 1 knowledge base only in the paper

#%% 0. Import libraries
# !pip install bs4 chromadb tiktoken faiss-cpu accelerate xformers ragas

import os
import time
from datasets import Dataset
import pandas as pd



from rag_vectorstore import similarity_search_doc, multi_similarity_search_doc
from rag_ragas import retriever_evaluation

from LogSetup import logger
from rag_ragas import retriever_evaluation

from StipVectorStore import StipVectorStore
from StipKnowledgeBase import StipKnowledgeBase, StipHuggingFaceDataset, load_50_qa_dataset, load_suswiki, load_wikipedia
from StipEmbedding import StipEmbedding

#### 
FOLDER_PATH = "experiments/distance_metrics_comp/"

EMBED_MODEL = StipEmbedding("bge").embed_model
CHUNK_SIZE = 200
CHUNK_OVERLAP_SCALE = 0.1


#### Knowledge Base 
suswiki_docs = load_suswiki()
# suswiki_docs["documents"] = suswiki_docs["documents"][:100] # limit data for test
print(f'successful load data from {suswiki_docs["source"]}. It has source and documents keys')

#### Embedding

#### Split and create vectorstore

EUCLEDIAN = "l2"
COSINE = "cosine"
INNER_PRODUCT = "ip"
TOP_K = 3

## 1. faiss
faiss_vs = StipVectorStore("faiss")
# faiss_eucledian = faiss_vs.create_vectorstore(suswiki_docs, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP_SCALE, EUCLEDIAN)
# faiss_cosine = faiss_vs.create_vectorstore(suswiki_docs, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP_SCALE, COSINE)
# faiss_ip = faiss_vs.create_vectorstore(suswiki_docs, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP_SCALE, INNER_PRODUCT)


# ## 2. chroma
chroma_vs = StipVectorStore("chroma")
# chroma_eucledian = chroma_vs.create_vectorstore(suswiki_docs, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP_SCALE, EUCLEDIAN)
# chroma_cosine = chroma_vs.create_vectorstore(suswiki_docs, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP_SCALE, COSINE)
# chroma_ip = chroma_vs.create_vectorstore(suswiki_docs, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP_SCALE, INNER_PRODUCT)

### sanity check create VS: get similarity search
# sanity_check_chroma_eucledian  = chroma_eucledian.similarity_search_with_score("A/B testing", k=3)
# sanity_check_chroma_cosine = chroma_cosine.similarity_search_with_score("A/B testing", k=3)
# sanity_check_chroma_ip = chroma_ip.similarity_search_with_score("A/B testing", k=3)
# sanity_check_faiss = faiss_ip.similarity_search_with_score("A/B testing", k=3)


#### Load vectorstore
## faiss
faiss_data_eucledian = faiss_vs.load_vectorstore("vectorstores/db_faiss/sustainability-methods-wiki/bge-large-en-v1.5_200_0.1_l2")
faiss_data_cosine = faiss_vs.load_vectorstore("vectorstores/db_faiss/sustainability-methods-wiki/bge-large-en-v1.5_200_0.1_cosine")
faiss_data_ip = faiss_vs.load_vectorstore("vectorstores/db_faiss/sustainability-methods-wiki/bge-large-en-v1.5_200_0.1_ip")
## chroma
chroma_data_eucledian = chroma_vs.load_vectorstore("vectorstores/db_chroma/sustainability-methods-wiki/bge-large-en-v1.5_200_0.1_l2")
chroma_data_cosine = chroma_vs.load_vectorstore("vectorstores/db_chroma/sustainability-methods-wiki/bge-large-en-v1.5_200_0.1_cosine")
chroma_data_ip = chroma_vs.load_vectorstore("vectorstores/db_chroma/sustainability-methods-wiki/bge-large-en-v1.5_200_0.1_ip")

### sanity check: get similarity search
# sanity_check_load_faiss_eucledian  = faiss_data_eucledian.similarity_search_with_score("A/B testing", k=3)
# sanity_check_load_faiss_cosine = faiss_data_cosine.similarity_search_with_score("A/B testing", k=3)
# sanity_check_load_faiss_ip = faiss_data_ip.similarity_search_with_score("A/B testing", k=3)
# sanity_check_load_chroma_eucledian  = chroma_data_eucledian.similarity_search_with_score("A/B testing", k=3)
# sanity_check_load_chroma_cosine = chroma_data_cosine.similarity_search_with_score("A/B testing", k=3)
# sanity_check_load_chroma_ip = chroma_data_ip.similarity_search_with_score("A/B testing", k=3)

#%% Questions dataset
curated_qa_dataset = load_50_qa_dataset()
dataset = curated_qa_dataset['train']

def check_is_different_contexts(datasets):
    """
    check if there are different contexts in the dataset
    """

    # convert datasets to pandas and apply tuple to contexts
    dfs = [dataset1.to_pandas() for dataset1 in datasets]
    contexts_tuples = [df["contexts"].apply(tuple) for df in dfs]

    # find rows with different contexts
    is_different = []
    for i in range(len(contexts_tuples[0])):  # assuming all DataFrames have the same number of rows
        row_values = [df.iloc[i] for df in contexts_tuples]
        is_different.append(not all(val == row_values[0] for val in row_values[1:]))

    return dfs, is_different

def find_different_contexts(*datasets):
    """
    find the different contexts between multiple datasets
    input: contexted_datasets
    output: dataframe with columns: question, ground_truths, contexts_1, contexts_2, ...
    """
    dfs, is_different = check_is_different_contexts(datasets)
        
    # select different rows and reset index
    different_dfs = [df[is_different].reset_index(drop=True) for df in dfs]

    # Select the desired columns from the first DataFrame
    first_df = different_dfs[0][["question", "ground_truths", "contexts"]]

    # Select the "contexts" column from the other DataFrames
    other_dfs = [df["contexts"].rename(f"contexts_{i+2}") for i, df in enumerate(different_dfs[1:])]

    # Concatenate these selections
    combined_df = pd.concat([first_df] + other_dfs, axis=1)

    return combined_df



#%% FAISS Vector store

faiss_eucledian_contexted = multi_similarity_search_doc(faiss_data_eucledian, dataset, TOP_K) 
faiss_cosine_contexted = multi_similarity_search_doc(faiss_data_cosine, dataset, TOP_K)
faiss_ip_contexted = multi_similarity_search_doc(faiss_data_ip, dataset, TOP_K)

chroma_eucledian_contexted = multi_similarity_search_doc(chroma_data_eucledian, dataset, TOP_K)
chroma_cosine_contexted = multi_similarity_search_doc(chroma_data_cosine, dataset, TOP_K)
chroma_ip_contexted = multi_similarity_search_doc(chroma_data_ip, dataset, TOP_K)

faiss_combined_df = find_different_contexts(faiss_eucledian_contexted, faiss_cosine_contexted, faiss_ip_contexted)
chroma_combined_df = find_different_contexts(chroma_eucledian_contexted, chroma_cosine_contexted, chroma_ip_contexted)

## export to csv
# faiss_combined_df.to_csv(FOLDER_PATH+"combined_df_faiss.csv", index=False)
# faiss_combined_df.to_json(FOLDER_PATH+"combined_df_faiss.json", orient="records")
# chroma_combined_df.to_csv(FOLDER_PATH+"combined_df_chroma.csv", index=False)
# chroma_combined_df.to_json(FOLDER_PATH+"combined_df_chroma.json", orient="records")

#%% RAGAS evaluate retriever
## only evaluate chroma, because faiss returned same results with different distance metrics

## evaluate chroma chroma_combined_df 3x. 
## first, evaluate the eucledian distance, columns used: question, ground_truths, contexts
## second, evaluate the cosine distance, columns used: question, ground_truths, contexts_2
## third, evaluate the inner product distance, columns used: question, ground_truths, contexts_3

diff_euc_chroma = Dataset.from_pandas(chroma_combined_df[['question', 'ground_truths', 'contexts']])
diff_cos_chroma = Dataset.from_pandas(chroma_combined_df[['question', 'ground_truths', 'contexts_2']].rename(columns={'contexts_2':'contexts'}))
diff_ip_chroma = Dataset.from_pandas(chroma_combined_df[['question', 'ground_truths', 'contexts_3']].rename(columns={'contexts_3':'contexts'}))


diff_euc_evaluated = retriever_evaluation(diff_euc_chroma, FOLDER_PATH+"eval_chroma_eucledian.csv")
diff_cos_evaluated = retriever_evaluation(diff_cos_chroma, FOLDER_PATH+"eval_chroma_cosine.csv")
diff_ip_evaluated = retriever_evaluation(diff_ip_chroma, FOLDER_PATH+"eval_chroma_ip.csv")

mean_context_precision_euc = diff_euc_evaluated['context_precision'].mean()
mean_context_precision_cos = diff_cos_evaluated['context_precision'].mean()
mean_context_precision_ip = diff_ip_evaluated['context_precision'].mean()

mean_context_recall_euc = diff_euc_evaluated['context_recall'].mean()
mean_context_recall_cos = diff_cos_evaluated['context_recall'].mean()
mean_context_recall_ip = diff_ip_evaluated['context_recall'].mean()

#calculate f1-measure
mean_context_f1_euc = 2 * (mean_context_precision_euc * mean_context_recall_euc) / (mean_context_precision_euc + mean_context_recall_euc)
mean_context_f1_cos = 2 * (mean_context_precision_cos * mean_context_recall_cos) / (mean_context_precision_cos + mean_context_recall_cos)
mean_context_f1_ip = 2 * (mean_context_precision_ip * mean_context_recall_ip) / (mean_context_precision_ip + mean_context_recall_ip)

print("")

#%%