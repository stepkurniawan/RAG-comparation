#%% Imports
from dotenv import load_dotenv
import os
load_dotenv()

import matplotlib.pyplot as plt
import time

from rag_embedding import get_embed_model, embedding_ids
from rag_vectorstore import get_index_vectorstore_wiki_nyc, similarity_search_doc, multi_similarity_search_doc, create_chroma_db, load_chroma_db, create_faiss_db
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

def compare_text_splitter(embed_id, top_k): 
    # for each chunk size, split the data (document), and create a vectorstore
    for chunk_size in chuck_sizes_list[:]:
        print(f"Testing chunk_size: {chunk_size}")
        test_suswiki_documents = load_sustainability_wiki_langchain_documents()
        test_docs = split_data_to_docs(test_suswiki_documents, chunk_size)
        print(f"Number of docs: {len(test_docs)}")

        # create vectorstore
        embed_model , _ = get_embed_model(embed_id)
        vectorstore = create_faiss_db(test_docs, embed_model, chunk_size)

        # # test retriving 1 question
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

        print("Evaluating context precision...")
        similar_context = context_precision.score(similar_context)
        print("Evaluating context recall...")
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


# %% CHECK THE CSV ##########################################

def sanity_check_text_splitter_results(embed_id, top_k):
    # create a dict based on the {chunk_size : avg_scores}
    avg_scores_dict = {}
    for chunk_size in chuck_sizes_list:
        file_path = f"./data/text_splitter_eval/retriever_evaluation_{chunk_size}_{embed_id}_{top_k}"
        df = pd.read_csv(f"{file_path}.csv", sep="|")
        df['avg_precision_recall'] = df[['context_precision', 'context_recall']].mean(axis=1)
        avg_score = df['avg_precision_recall'].mean()
        avg_scores_dict[chunk_size] = avg_score
    
    # create_bar_chart(avg_scores_dict)
    create_line_chart(df, avg_scores_dict)


    
    

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

def create_line_chart(df, avg_scores_dict):
    # create a plot that consists of 2 line charts. 
    # 1 line chart from 'context_precision', and the other 'context_recall'. Dont forget the legend and use different color
    # the x axis is the chunk_size (avg_scores_dict's keys), the y axis is the avg_score avg_scores_dict's values
    plt.plot(list(avg_scores_dict.keys()), df['context_precision'], label='context_precision')
    plt.plot(list(avg_scores_dict.keys()), df['context_recall'], label='context_recall')
    plt.plot(list(avg_scores_dict.keys()), df['avg_precision_recall'], label='avg_precision_recall')
    plt.legend()
    plt.show()






# %%
start_time = time.time()


compare_text_splitter(embedding_ids['BGE_LARGE_ID'], 1)
# sanity_check_text_splitter_results(embedding_ids['BGE_LARGE_ID'], 1)


end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds, or {(end_time - start_time)/60} minutes, or {(end_time - start_time)/3600} hours")
