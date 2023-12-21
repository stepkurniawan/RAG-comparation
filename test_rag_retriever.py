#%% Imports
from dotenv import load_dotenv
import os
load_dotenv()

import matplotlib.pyplot as plt
import time

from rag_embedding import get_embed_model, embedding_ids
from rag_vectorstore import get_index_vectorstore_wiki_nyc, similarity_search_doc, multi_similarity_search_doc, create_chroma_db, load_chroma_db, create_faiss_db, create_db_pipeline
from rag_load_data import get_arxiv_data_from_dataset, load_from_webpage, load_sustainability_wiki_dataset, load_sustainability_wiki_langchain_documents
from rag_splitter import split_data_to_docs, chuck_sizes_list
from rag_llms import load_llm_tokenizer_hf_with_model
from rag_llms import LLAMA2_13B_CHAT_MODEL_ID
from rag_load_data import load_50_qa_dataset

from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS, Chroma
from langchain.docstore.document import Document
from datasets import load_dataset, Dataset, DatasetDict

from ragas.metrics import ContextPrecision, ContextRecall

import pandas as pd

embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
QUERY ="What is the probability of you being so much taller than the average? "
DB_PATH = "vectorstores/db_faiss"
# embedding_ids
chunk_overlap_scale_list = [0.1, 0.25, 0.5, 0.75]


#%% LOAD DATA ##########################################

def test_load_from_webpage():
    link = "https://sustainabilitymethods.org/index.php/A_matter_of_probability"
    data = load_from_webpage(link) # its a list(document)
    assert isinstance(data, list) , "Failed loading data from webpage, check load_from_webpage()"
    return data

test_data=test_load_from_webpage()
print(test_data)

# %% 
def test_load_suswiki_dataset():
    test_suswiki_dataset = load_sustainability_wiki_dataset()
    assert isinstance(test_suswiki_dataset, Dataset) , "Failed loading sustainability wiki dataset, check load_sustainability_wiki_dataset()"
    return test_suswiki_dataset

test_suswiki_dataset=test_load_suswiki_dataset()
print(test_suswiki_dataset[0])

# %%
def test_load_sustainability_wiki_langchain_documents():
    test_suswiki_dataset = load_sustainability_wiki_langchain_documents()
    assert isinstance(test_suswiki_dataset, list) , "Failed loading sustainability wiki dataset, check load_sustainability_wiki_dataset()"
    assert isinstance(test_suswiki_dataset[0], Document) , "Failed loading sustainability wiki dataset, it is not type Document Object, check load_sustainability_wiki_dataset()"
    assert isinstance(test_suswiki_dataset[0].page_content, str) , "Failed loading sustainability wiki dataset, page_content is not string , check load_sustainability_wiki_dataset()"
    return test_suswiki_dataset

test_suswiki_documents = test_load_sustainability_wiki_langchain_documents()
print(test_suswiki_documents[0].page_content)

# %% SPLITTING DATA ##########################################

def test_single_split_data_to_docs(test_data):
    docs = split_data_to_docs(test_data)
    assert isinstance(docs, list) , "Failed splitting data to docs, check split_data_to_docs()"
    assert isinstance(docs[0], Document) , "Failed splitting data to docs, it is not type Document Object, check split_data_to_docs()"
    assert isinstance(docs[0].page_content, str) , "Failed splitting data to docs, page_content is not string , check split_data_to_docs()"
    return docs

test_docs=test_single_split_data_to_docs(test_data)
print(test_docs[0].page_content)

# all the suswiki documents are getting splitted based on the text chunks, and stored as list(Document), and each document has page_content attribute which is the chunk in strings
test_docs=test_single_split_data_to_docs(test_suswiki_documents)
print(test_docs[55].page_content) 





# %% 
def test_get_embed_model(embed_model_id):
    embed_model, embed_model_name = get_embed_model(embed_model_id)
    # assert isinstance(embed_model, HuggingFaceEmbeddings) , "Failed getting the embedding model, check get_embed_model()"
    # assert embed_model.model_name == embed_model_id
    return embed_model

test_embed_model = test_get_embed_model(embed_model_id)

for emb_name in embedding_ids:
    print(f"Testing embedding: {emb_name}")
    print(f"Embedding id: {embedding_ids[emb_name]}")
    test_embed_model = test_get_embed_model(embedding_ids[emb_name])

    
# %% Testing the embedding

def test_print_doc_dimensions(): 
    docs = [
        "this is one document",
        "and another document"
    ]

    embeddings = test_embed_model.embed_documents(docs)

    print(f"We have {len(embeddings)} doc embeddings, each with "
        f"a dimensionality of {len(embeddings[0])}.")

test_print_doc_dimensions()
#%% VECTOR STORE

def test_get_index_vectorstore_wiki_nyc(embed_model):
    index = get_index_vectorstore_wiki_nyc(embed_model)
    # assert isinstance(index, VectorstoreIndexWrapper) , "Failed getting the index, check get_index_vectorstore_wiki_nyc()"
    assert isinstance(index.vectorstore, FAISS) , "Failed getting the index, check get_index_vectorstore_wiki_nyc()"
    return index

test_vectorstore_faiss_index = test_get_index_vectorstore_wiki_nyc(test_embed_model)

#%%

def test_create_chroma_db(test_data):
    #requirement
    test_docs=test_single_split_data_to_docs(test_data)

    #test
    vectorstore = create_chroma_db(test_docs, test_embed_model)
    assert isinstance(vectorstore, Chroma) , "Failed creating the chroma db, check create_chroma_db()"

test_create_chroma_db(test_data)

#%%

def test_load_chroma_db():
    vectorstore = load_chroma_db(test_embed_model)
    assert isinstance(vectorstore, Chroma) , "Failed loading the chroma db, check load_chroma_db()"
    return vectorstore

vectorstore = test_load_chroma_db()

# %%

def test_similarity_search():
    embed_model = test_embed_model
    db = FAISS.load_local(DB_PATH, embed_model)
    query = QUERY

    similar_context = similarity_search_doc(db, query)
    assert isinstance(similar_context, list) , "Failed getting the similar context, check similarity_search()"

# similar_context = test_similarity_search()




# %% TEST PIPELINE ##########################################

llm = load_llm_tokenizer_hf_with_model(LLAMA2_13B_CHAT_MODEL_ID) 
embed_model , _ = get_embed_model(embedding_ids['BGE_LARGE_ID'])
# db = load_local_faiss_vector_database(embed_model)
# qa_dataset = load_50_qa_dataset()
# qa_dataset = qa_dataset['train'][:3] # if you limit using [:2] -> it will become a dictionary, not a dataset


# %% FINDING THE BEST TEXT SPLIT ##########################################
# TODO output: graph with X axis: chunk_size, Y axis: context recall, precision, and F-measure

def compare_text_splitter(embed_id, top_k, chunk_overlap_scale=0.1): 
    # for each chunk size, split the data (document), and create a vectorstore
    
    # also measure the time taken to split the data, and search the similarity
    
    time_df = pd.DataFrame(columns=['chunk_size', 'split_time', 'create_vector_time' ,'search_time', 'total_time']) # dataframe to note the time

    for chunk_size in chuck_sizes_list[:]:
        print(f"Testing chunk_size: {chunk_size}")
        test_suswiki_documents = load_sustainability_wiki_langchain_documents()
        # split docs
        start_split_time = time.time()
        test_docs = split_data_to_docs(test_suswiki_documents, chunk_size, chunk_overlap_scale)
        print(f"Number of docs for {chunk_size} chunk_size is: {len(test_docs)}")
        end_split_time = time.time()
        split_time = end_split_time - start_split_time
        print(f"Time taken to split: {split_time} seconds, or {split_time/60} minutes, or {split_time/3600} hours")

        # create vectorstore
        embed_model , _ = get_embed_model(embed_id)
        start_create_vector_time = time.time()

        vectorstore = create_faiss_db(test_docs, embed_model, chunk_size)
        
        end_create_vector_time = time.time()
        create_vector_time = end_create_vector_time - start_create_vector_time
        print(f"Time taken to create vectorstore: {create_vector_time} seconds, or {create_vector_time/60} minutes, or {create_vector_time/3600} hours")

        # # test retriving 1 question
        # sample_query = "what is Generalized Linear Models?"
        # sample_similar_context = similarity_search_doc(vectorstore, sample_query, 1)
        # print(sample_similar_context)

        # load questions
        qa_dataset = load_50_qa_dataset()

        # get similarity search
        start_search_time = time.time()
        similar_context = multi_similarity_search_doc(vectorstore, qa_dataset, top_k)
        end_search_time = time.time()
        search_time = end_search_time - start_search_time
        print(f"Time taken to search: {search_time} seconds, or {search_time/60} minutes, or {search_time/3600} hours") # for 50 questions
        single_search_time = search_time/50
        print(f"Time taken to search 1 question: {single_search_time} seconds, or {single_search_time/60} minutes, or {single_search_time/3600} hours") # for 1 question

        total_time = split_time + create_vector_time + search_time
        print(f"Total time taken: {total_time} seconds, or {total_time/60} minutes, or {total_time/3600} hours")

        # add the time to the dataframe
        time_df = pd.concat([time_df, pd.DataFrame({'chunk_size': [chunk_size], 'split_time': [split_time], 'create_vector_time': [create_vector_time], 'search_time': [search_time], 'total_time': [total_time]})])

        #### RAGAS EVALUATION ##############################
        context_precision = ContextPrecision()
        context_recall = ContextRecall(batch_size=10)
        similar_context = context_precision.score(similar_context)
        similar_context = context_recall.score(similar_context)

        similar_context_df = pd.DataFrame(similar_context)
        # save the answer into a csv
        print("creating csv with chunk_size: ", chunk_size)

        # Check if the directory exists, if not, create it
        file_path = f"./data/text_splitter_eval/retriever_evaluation_{chunk_size}_{embed_id}_{top_k}"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        similar_context_df.to_csv(file_path +".csv", sep="|", )

        #save answer to json
        print("creating json with chunk_size: ", chunk_size)
        similar_context_df.to_json(file_path+".json")

    # save the time_df into a csv
    print("creating csv with time_df")
    time_df.to_csv(f"./data/text_splitter_eval/time_df_{len(test_docs)}.csv", sep="|", )



####  COMPARE TEXT OVERLAP ################

# im using chunksize of 400 because i want to see the effect of the overlap better. 
# using too little will hinder any effect of the overlap
# testing 10%, 25%, 50%, 75%, 100%, 200% overlap


def compare_text_overlap(embed_id, top_k, chunk_size=400):
    chunk_size=400
    # for each chunk_overlap_scale_list, split the data (document), and create a vectorstore
    print(f"Testing chunk_size: {chunk_size}")

    for chunk_overlap_scale in chunk_overlap_scale_list:
        vectorstore = create_db_pipeline("suswiki", "FAISS", embed_id, chunk_size, chunk_overlap_scale)

        # # # # test retrieving 1 question
        # sample_query = "what is Generalized Linear Models?"
        # sample_similar_context = similarity_search_doc(vectorstore, sample_query, 1)
        # print(sample_similar_context)

        # load questions
        qa_dataset = load_50_qa_dataset()

        # get similarity search
        similar_context = multi_similarity_search_doc(vectorstore, qa_dataset, top_k)

        # answer using ragas evaluate
        context_precision = ContextPrecision()
        context_recall = ContextRecall(batch_size=10)
        similar_context = context_precision.score(similar_context)
        similar_context = context_recall.score(similar_context)

        similar_context_df = pd.DataFrame(similar_context)
        # save the answer into a csv
        print("creating csv with chunk_overlap_scale: ", chunk_overlap_scale)

        # Check if the directory exists, if not, create it
        name_embed_id = embed_id.split('/')[-1]
        file_path = f"./data/text_overlap_eval/chunk_overlap_evaluation_{chunk_overlap_scale}_{name_embed_id}_{top_k}"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        similar_context_df.to_csv(file_path +".csv", sep="|", )

        #save answer to json
        similar_context_df.to_json(file_path+".json")






    


#  CHECK THE CSV ##########################################

def sanity_check_text_splitter_results(embed_id, top_k, chunk_overlap=0.1):
    # create a dict based on the {chunk_size : avg_scores}
    avg_scores_dict = {}
    df_line = pd.DataFrame(columns=['chunk_size', 'mean_context_precision', 'mean_context_recall'])
    for chunk_size in chuck_sizes_list:
        file_path = f"./data/text_splitter_eval/retriever_evaluation_{chunk_size}_{embed_id}_{top_k}"
        df = pd.read_csv(f"{file_path}.csv", sep="|")
        df['avg_precision_recall'] = df[['context_precision', 'context_recall']].mean(axis=1)
        avg_score = df['avg_precision_recall'].mean()
        avg_scores_dict[chunk_size] = avg_score

        # create a dataframe for line chart. The columns are chunk_size, mean_context_precision, mean_context_recall
        # mean_context_precision is from all the context_precision in the df divided by the number of rows
        # mean_context_recall is from all the context_recall in the df divided by the number of rows
        # append the df_line with the chunk_size, mean_context_precision, mean_context_recall
        mean_context_precision = df['context_precision'].mean()
        mean_context_recall = df['context_recall'].mean()
        f_measure = 2 * (mean_context_precision * mean_context_recall) / (mean_context_precision + mean_context_recall)
        df_line = pd.concat([df_line, pd.DataFrame({'chunk_size': [chunk_size], 'mean_context_precision': [mean_context_precision], 'mean_context_recall': [mean_context_recall], 'f_measure': [f_measure]})])
    
    df_line = df_line.reset_index(drop=True)  # fix the index 

    # create_bar_chart(avg_scores_dict)
    create_line_chart(df_line)

    # add title
    # plt.text(0.5, 0.90, f'embed_id: {embed_id}, top_k: {top_k}', ha='center', va='center', transform=plt.gcf().transFigure)
    plt.title(f'embed_id: {embed_id}, top_k: {top_k}')
    
    # add information below graphs
    # plt.text(0.5, -0.01, "chunk_overlap: 10%", ha='center', va='center', transform=plt.gcf().transFigure)




##### now, another sanity check, but for chunk_overlap #####################

def sanity_check_text_overlap_results(embed_id, top_k, chunk_size=400):
    # create a dict based on the {chunk_overlap : avg_scores}
    avg_scores_dict = {}
    df_line = pd.DataFrame(columns=['chunk_overlap', 'mean_context_precision', 'mean_context_recall'])
    name_embed_id = embed_id.split('/')[-1]
    for chunk_overlap_scale in chunk_overlap_scale_list:
        file_path = f"./data/text_overlap_eval/chunk_overlap_evaluation_{chunk_overlap_scale}_{name_embed_id}_{top_k}"
        df = pd.read_csv(f"{file_path}.csv", sep="|")
        df['avg_precision_recall'] = df[['context_precision', 'context_recall']].mean(axis=1)
        avg_score = df['avg_precision_recall'].mean()
        avg_scores_dict[chunk_overlap_scale] = avg_score

        # create a dataframe for line chart. The columns are chunk_size, mean_context_precision, mean_context_recall
        # mean_context_precision is from all the context_precision in the df divided by the number of rows
        # mean_context_recall is from all the context_recall in the df divided by the number of rows
        # append the df_line with the chunk_size, mean_context_precision, mean_context_recall
        mean_context_precision = df['context_precision'].mean()
        mean_context_recall = df['context_recall'].mean()
        f_measure = 2 * (mean_context_precision * mean_context_recall) / (mean_context_precision + mean_context_recall)
        df_line = pd.concat([df_line, pd.DataFrame({'chunk_overlap': [chunk_overlap_scale], 'mean_context_precision': [mean_context_precision], 'mean_context_recall': [mean_context_recall], 'f_measure': [f_measure]})])

    df_line = df_line.reset_index(drop=True)  # fix the index

    



def create_bar_chart(avg_scores_dict):    
    # create a bar chart based on the dict, the x axis is the chunksize, the y axis is the avg_score
    # bar chart: the x axis is the chunksize (avg_scores_dict's keys), the y axis is the avg_score avg_scores_dict's values
    plt.bar(range(len(avg_scores_dict)), list(avg_scores_dict.values()), align='center')
    #change the label of the x axis to be chunksize 
    plt.xticks(range(len(avg_scores_dict)), list(avg_scores_dict.keys()))

    # give labels
    plt.xlabel('chunk_size')
    plt.ylabel('avg_score')
    plt.show()



def create_line_chart(df_line):
    # IMPORTANT: the chunk_size or the chunk_overlap_scale must be the first column

    plt.plot(df_line.iloc[:, 0], df_line['mean_context_precision'], label='mean_context_precision', alpha=0.5)

    plt.plot(df_line.iloc[:, 0], df_line['mean_context_recall'], label='mean_context_recall', alpha=0.5)
    plt.plot(df_line.iloc[:, 0], df_line['f_measure'], label='f_measure', linewidth=5)
    plt.xlabel(df_line.columns[0])
    plt.ylabel('score')
    plt.legend()
    # annotate the score on each of the point
    for i in range(len(df_line)):
        plt.annotate(round(df_line['mean_context_precision'][i], 3), (df_line.iloc[:, 0][i], df_line['mean_context_precision'][i]),
                     alpha=0.5)
        plt.annotate(round(df_line['mean_context_recall'][i], 3), (df_line.iloc[:, 0][i], df_line['mean_context_recall'][i]),
                     alpha=0.5)
        plt.annotate(round(df_line['f_measure'][i], 4), (df_line.iloc[:, 0][i], df_line['f_measure'][i]))

    plt.xticks(df_line.iloc[:, 0])
    plt.show()



########## RUN THE TEST ############################
start_time = time.time()


# compare_text_splitter(embedding_ids['BGE_LARGE_ID'], 1, 0.1)
# sanity_check_text_splitter_results(embedding_ids['BGE_LARGE_ID'], 1)

## compare overlap

# compare_text_overlap(embedding_ids['BGE_LARGE_ID'], 1, 400)
sanity_check_text_overlap_results(embedding_ids['BGE_LARGE_ID'], 1)


end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds, or {(end_time - start_time)/60} minutes, or {(end_time - start_time)/3600} hours")

# %%
