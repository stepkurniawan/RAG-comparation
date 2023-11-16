#%% Imports
from dotenv import load_dotenv
import os
load_dotenv()
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
print(test_docs[0].page_content) 





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

similar_context = test_similarity_search()




# %% TEST PIPELINE ##########################################

llm = load_llm_tokenizer_hf_with_model(LLAMA2_13B_CHAT_MODEL_ID) 
embed_model , _ = get_embed_model(embedding_ids['BGE_LARGE_ID'])
# db = load_local_faiss_vector_database(embed_model)
# qa_dataset = load_50_qa_dataset()
# qa_dataset = qa_dataset['train'][:3] # if you limit using [:2] -> it will become a dictionary, not a dataset


# %% FINDING THE BEST TEXT SPLIT ##########################################
# TODO output: graph with X axis: chunk_size, Y axis: context recall, precision, and F-measure

# for each chunk size, split the data (document), and create a vectorstore
for chunk_size in chuck_sizes_list[:]:
    print(f"Testing chunk_size: {chunk_size}")
    test_docs = split_data_to_docs(test_data, chunk_size)
    print(f"Number of docs: {len(test_docs)}")

    # create vectorstore
    init_embed_id = embedding_ids['BGE_LARGE_ID']
    embed_model , _ = get_embed_model(init_embed_id)
    vectorstore = create_faiss_db(test_docs, embed_model)

    # load questions
    qa_dataset = load_50_qa_dataset()

    # get similarity search
    similar_context = multi_similarity_search_doc(vectorstore, qa_dataset, 1)

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
    similar_context_df.to_csv(f"data/retriever_evaluation_{chunk_size}.csv", sep="|", )

    #save answer to json
    print("creating json with chunk_size: ", chunk_size)
    similar_context_df.to_json(f"data/retriever_evaluation_{chunk_size}.json")

    # TODO: check this value and create a graph

