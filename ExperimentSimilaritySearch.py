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

import pandas as pd

from rag_vectorstore import similarity_search_doc, multi_similarity_search_doc

from LogSetup import logger
from rag_ragas import retriever_evaluation

from StipVectorStore import StipVectorStore
from StipKnowledgeBase import StipKnowledgeBase, StipHuggingFaceDataset
from StipEmbedding import StipEmbedding

#### 
EMBED_MODEL = StipEmbedding("bge").embed_model
CHUNK_SIZE = 200
CHUNK_OVERLAP_SCALE = 0.1


#### Knowledge Base 
suswiki_kb = StipKnowledgeBase("stepkurniawan/sustainability-methods-wiki", None, "huggingface_cache/suswiki_hf")
suswiki_docs = suswiki_kb.load_documents()
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
curated_qa_dataset = StipHuggingFaceDataset("stepkurniawan/sustainability-methods-wiki", "50_QA_reviewed", ["question", "ground_truths"]).load_dataset()
dataset = curated_qa_dataset['train']


def find_different_contexts(*datasets):
    """
    find the different contexts between multiple datasets
    input: contexted_datasets
    output: dataframe with columns: question, ground_truths, contexts_1, contexts_2, ...
    """

    # convert datasets to pandas and apply tuple to contexts
    dfs = [dataset.to_pandas() for dataset in datasets]
    contexts_tuples = [df["contexts"].apply(tuple) for df in dfs]

    # find rows with different contexts
    is_different = []
    for i in range(len(contexts_tuples[0])):  # assuming all DataFrames have the same number of rows
        row_values = [df.iloc[i] for df in contexts_tuples]
        is_different.append(not all(val == row_values[0] for val in row_values[1:]))
        
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

faiss_combined_df_eucledian = find_different_contexts(faiss_eucledian_contexted, faiss_cosine_contexted, faiss_ip_contexted)
chroma_combined_df_eucledian = find_different_contexts(chroma_eucledian_contexted, chroma_cosine_contexted, chroma_ip_contexted)

# export to csv
folder_path = "experiments/distance_metrics_comp/"
faiss_combined_df_eucledian.to_csv(folder_path+"combined_df_faiss.csv", index=False)
faiss_combined_df_eucledian.to_json(folder_path+"combined_df_faiss.json", orient="records")
chroma_combined_df_eucledian.to_csv(folder_path+"combined_df_chroma.csv", index=False)
chroma_combined_df_eucledian.to_json(folder_path+"combined_df_chroma.json", orient="records")
print("")

#%% 