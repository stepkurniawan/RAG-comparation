#%%### prepare environement
import os
import time
import pandas as pd
from rag_vectorstore import similarity_search_doc, multi_similarity_search_doc
from rag_ragas import retriever_evaluation
from LogSetup import logger
from rag_ragas import retriever_evaluation, load_retriever_evaluation
from StipVectorStore import StipVectorStore
from StipKnowledgeBase import StipKnowledgeBase, load_suswiki, load_wikipedia, load_50_qa_dataset
from StipEmbedding import StipEmbedding

####
EMBED_MODEL = StipEmbedding("bge").embed_model
CHUNK_SIZE = 200
CHUNK_OVERLAP_SCALE = 0.1
TOP_K = 3
INDEX_DISTANCE = "l2"
VECTORSTORE = StipVectorStore("faiss")
QUESTION_DATASET = load_50_qa_dataset()['train']
FOLDER_PATH ="experiments/KnowledgeBase/"


#%%### retriever
#### create vector store (ONLY FIRST TIME) #### 
## load knowledge base
# suswiki_kb = load_suswiki()
# wikipedia_kb = load_wikipedia()
## create vector store
# suswiki_vectorstore = VECTORSTORE.create_vectorstore(suswiki_kb, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP_SCALE, INDEX_DISTANCE).db
# wikipedia_vectorstore = VECTORSTORE.create_vectorstore(wikipedia_kb, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP_SCALE, INDEX_DISTANCE).db

#### load vector store
suswiki_vectorstore = VECTORSTORE.load_vectorstore("vectorstores/db_faiss/sustainability-methods-wiki/bge-large-en-v1.5_200_0.1_l2")
wikipedia_vectorstore = VECTORSTORE.load_vectorstore("vectorstores/db_faiss/wikipedia/bge-large-en-v1.5_200_0.1_l2")
#%% 

## sanity check
sanity_suswiki = suswiki_vectorstore.similarity_search_with_score("A/B testing", k=3)
sanity_wikipedia = wikipedia_vectorstore.similarity_search_with_score("A/B testing", k=3)

# %% search for context

contexted_suswiki = multi_similarity_search_doc(suswiki_vectorstore, QUESTION_DATASET, TOP_K)
contexted_wikipedia = multi_similarity_search_doc(wikipedia_vectorstore, QUESTION_DATASET, TOP_K)


#%% evaluate using RAGAS
suswiki_eval = retriever_evaluation(contexted_suswiki, FOLDER_PATH+"suswiki_eval2.csv")
wikipedia_eval = retriever_evaluation(contexted_wikipedia, FOLDER_PATH+"wikipedia_eval2.csv")

suswiki_eval = load_retriever_evaluation(FOLDER_PATH+"suswiki_eval2.csv")
wikipedia_eval = load_retriever_evaluation(FOLDER_PATH+"wikipedia_eval2.csv")

print("")
## calculate mean precision, recall and F1 score

def calculate_mean_score(df):
    mean_precision = df['context_precision'].mean()
    mean_recall = df['context_recall'].mean()
    f1 = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall)
    return mean_precision, mean_recall, f1

suswiki_mean_precision, suswiki_mean_recall, suswiki_f1 = calculate_mean_score(suswiki_eval)
wikipedia_mean_precision, wikipedia_mean_recall, wikipedia_f1 = calculate_mean_score(wikipedia_eval)

# combines the results into 1 dataframe
final_df = pd.DataFrame(columns=['Knowledge Base', 'Mean Precision', 'Mean Recall', 'F1 Score'])
final_df = pd.concat([final_df, pd.DataFrame([{'Knowledge Base': 'suswiki', 'Mean Precision': suswiki_mean_precision, 'Mean Recall': suswiki_mean_recall, 'F1 Score': suswiki_f1}])], ignore_index=True)
final_df = pd.concat([final_df, pd.DataFrame([{'Knowledge Base': 'wikipedia', 'Mean Precision': wikipedia_mean_precision, 'Mean Recall': wikipedia_mean_recall, 'F1 Score': wikipedia_f1}])], ignore_index=True)

# %% additional statistics about the knowledge base

# number of datapoints 
suswiki_vs = VECTORSTORE.load_vectorstore("vectorstores/db_faiss/sustainability-methods-wiki/bge-large-en-v1.5_200_0.1_l2")
wikipedia_vs = VECTORSTORE.load_vectorstore("vectorstores/db_faiss/wikipedia/bge-large-en-v1.5_200_0.1_l2")
number_of_datapoints_suswiki = suswiki_vs.ndata
number_of_datapoints_wikipedia = wikipedia_vs.ndata
# %%


#%% explore wikipedia dataset from huggingface
from datasets import load_dataset

wikipedia_kb = load_dataset('wikipedia', '20220301.simple', split='train')
wikipedia_dataset = wikipedia_kb.select_columns(['title', 'text'])
print("success loading wikipedia dataset from HF")
print("the first title is: ", wikipedia_dataset['title'][0])
print("the first text is: ", wikipedia_dataset['text'][0])

# how many rows? 
print("the number of rows in train dataset is: ", wikipedia_dataset.num_rows) # 205 328

# how many total characters in the text column?
total_characters = 0
for text in wikipedia_dataset['text']:
    total_characters += len(text)
print("the total characters in the text column is: ", total_characters) # 215 489 882

# how many total words in the text column?
total_words = 0
for text in wikipedia_dataset['text']:
    total_words += len(text.split())
print("the total words in the text column is: ", total_words) # 34 489 908

# average number of words per text
avg_words = total_words / wikipedia_dataset.num_rows
print("the average number of words per text is: ", total_words / wikipedia_dataset.num_rows) # 167.97




