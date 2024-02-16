#%%### prepare environement
import os
import time
import pandas as pd
from datasets import load_dataset, Dataset
from rag_vectorstore import similarity_search_doc, multi_similarity_search_doc
from rag_ragas import retriever_evaluation, evaluate_qa_dataset_with_response, evaluate_qa_dataset_with_gen_df
from rag_chains import generate_context_answer_langchain, retrieval_qa_chain_from_local_db
from rag_llms import get_llama2_llm
from LogSetup import logger
from rag_ragas import retriever_evaluation, load_retriever_evaluation
from StipVectorStore import StipVectorStore
from StipKnowledgeBase import StipKnowledgeBase, load_suswiki, load_wikipedia, load_50_qa_dataset
from StipEmbedding import StipEmbedding


# %% ### prepare global variables
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
suswiki_vectorstore = VECTORSTORE.load_vectorstore("vectorstores/db_faiss/sustainability-methods-wiki/bge-large-en-v1.5_200_0.1_l2").db
wikipedia_vectorstore = VECTORSTORE.load_vectorstore("vectorstores/db_faiss/wikipedia/bge-large-en-v1.5_200_0.1_l2").db
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
suswiki_vs = VECTORSTORE.load_vectorstore("vectorstores/db_faiss/sustainability-methods-wiki/bge-large-en-v1.5_200_0.1_l2").db
wikipedia_vs = VECTORSTORE.load_vectorstore("vectorstores/db_faiss/wikipedia/bge-large-en-v1.5_200_0.1_l2").db
number_of_datapoints_suswiki = suswiki_vs.index.ntotal
number_of_datapoints_wikipedia = wikipedia_vs.index.ntotal
# %%


#%% explore wikipedia dataset from huggingface

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

## SUSWIKI
suswiki_kb = load_suswiki()
# how many rows? 
print("the number of rows in suswiki dataset is: ", len(suswiki_kb['documents'])) # 226
# how many total characters in the text column?
total_characters = 0
for text in [doc.page_content for doc in suswiki_kb['documents']]:
    total_characters += len(text)
print("the total characters in the text column is: ", total_characters) # 3 282 387
# how many total words in the text column?
total_words = 0
for text in [doc.page_content for doc in suswiki_kb['documents']]:
    total_words += len(text.split())
print("the total words in the text column is: ", total_words) # 484 395
# average number of words per text
avg_words = total_words / len(suswiki_kb['documents'])
print("the average number of words per text is: ", avg_words) # 2143.34


#%% GENERATION

language_model = get_llama2_llm()
k = 3
folder_path = "experiments/KnowledgeBase/"
folderpath_suswiki = folder_path+"suswiki"
folderpath_wikipedia = folder_path+"wikipedia"
if folder_path is not None:
    os.makedirs(folder_path, exist_ok=True)
if folderpath_suswiki is not None:
    os.makedirs(folderpath_suswiki, exist_ok=True)
if folderpath_wikipedia is not None:
    os.makedirs(folderpath_wikipedia, exist_ok=True)


suswiki_generate_df = generate_context_answer_langchain(QUESTION_DATASET, language_model, suswiki_vs, k, folder_save_path=folder_path+"suswiki_2_")
# wikipedia_generate_df = generate_context_answer_langchain(QUESTION_DATASET, language_model, wikipedia_vs, k, folder_save_path=folder_path+"wikipedia_2_")

# after generation, we can load it
suswiki_generate_df = pd.read_csv("experiments/KnowledgeBase/suswiki_2_llama2_FAISS_3_gen.csv")
wikipedia_generate_df = pd.read_csv("experiments/KnowledgeBase/wikipedia_2_llama2_FAISS_3_gen.csv")

#%% EVALUATION GENERATION

suswiki_evaldf = evaluate_qa_dataset_with_gen_df(suswiki_generate_df, QUESTION_DATASET, folder_path+"suswiki")
wikipedia_evaldf = evaluate_qa_dataset_with_response(wikipedia_generate_df, QUESTION_DATASET, folder_path+"wikipedia")

# suswiki_evaldf = pd.read_csv(folder_path+"suswiki_eval.csv")
# wikipedia_evaldf = pd.read_csv(folder_path+"wikipedia_eval2.csv")


# %%
from langchain.docstore.document import Document

from ragas.langchain.evalchain import RagasEvaluatorChain
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
faithfulness_chain = RagasEvaluatorChain(metric=faithfulness)
answer_rel_chain = RagasEvaluatorChain(metric=answer_relevancy)
context_rel_chain = RagasEvaluatorChain(metric=context_precision)
context_recall_chain = RagasEvaluatorChain(metric=context_recall)
ragas_metric = [faithfulness, answer_relevancy, context_precision, context_recall]


output_df = pd.DataFrame()
response_df = suswiki_generate_df
response_dict = response_df.to_dict() # list of keys: query, ground_truths, result, source_documents
response_dataset = Dataset.from_pandas(response_df)
amnesty_qa = load_dataset("explodinggradients/amnesty_qa", "english_v2")
qa_chain = retrieval_qa_chain_from_local_db(llm=language_model, vectorstore=suswiki_vectorstore, k=k)
result = qa_chain({"query": response_dict['query'][0]})

for i in range(0, len(QUESTION_DATASET["question"])):
    current_row_dict = response_df.iloc[i].to_dict()
    ## EVALUATION using RAGAS
    if faithfulness in ragas_metric:
        faithfulness_eval = faithfulness_chain(current_row_dict)
    if answer_relevancy in ragas_metric:
        answer_rel_eval = answer_rel_chain(current_row_dict)

    
# %%
