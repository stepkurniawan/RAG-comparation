#%% Imports

from rag_embedding import get_embed_model, embedding_ids
from rag_vectorstore import get_index_vectorstore_wiki_nyc, similarity_search_doc, create_chroma_db, load_chroma_db
from rag_load_data import get_arxiv_data_from_dataset, load_from_webpage, load_sustainability_wiki_dataset
from rag_splitter import split_data_to_docs


from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS, Chroma
from langchain.docstore.document import Document
from datasets import load_dataset, Dataset, DatasetDict


embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
QUERY ="What is the probability of you being so much taller than the average? "
DB_PATH = "vectorstores/db_faiss"



#%% LOAD DATA ##########################################

def test_load_from_webpage():
    link = "https://sustainabilitymethods.org/index.php/A_matter_of_probability"
    data = load_from_webpage(link)
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




# %% SPLITTING DATA ##########################################

def test_split_data_to_docs():
    docs = split_data_to_docs(test_data)
    assert isinstance(docs, list) , "Failed splitting data to docs, check split_data_to_docs()"
    assert isinstance(docs[0], Document) , "Failed splitting data to docs, it is not type Document Object, check split_data_to_docs()"
    assert isinstance(docs[0].page_content, str) , "Failed splitting data to docs, page_content is not string , check split_data_to_docs()"
    return docs

test_docs=test_split_data_to_docs()
print(test_docs[0].page_content)



# %% 
def test_get_embed_model(embed_model_id):
    embed_model, embed_model_name = get_embed_model(embed_model_id)
    # assert isinstance(embed_model, HuggingFaceEmbeddings) , "Failed getting the embedding model, check get_embed_model()"
    # assert embed_model.model_name == embed_model_id
    return embed_model

for emb_name in embedding_ids:
    print(f"Testing embedding: {emb_name}")
    print(f"Embedding id: {embedding_ids[emb_name]}")
    test_get_embed_model(embedding_ids[emb_name])
    
# %% Testing the embedding

def test_print_doc_dimensions(): 
    docs = [
        "this is one document",
        "and another document"
    ]

    embed_model = get_embed_model(embed_model_id)

    embeddings = embed_model.embed_documents(docs)

    print(f"We have {len(embeddings)} doc embeddings, each with "
        f"a dimensionality of {len(embeddings[0])}.")

test_print_doc_dimensions()
#%% VECTOR STORE

def test_get_index_vectorstore_wiki_nyc(embed_model):
    embed_model = get_embed_model(embed_model_id)
    index = get_index_vectorstore_wiki_nyc(embed_model)
    # assert isinstance(index, VectorstoreIndexWrapper) , "Failed getting the index, check get_index_vectorstore_wiki_nyc()"
    assert isinstance(index.vectorstore, FAISS) , "Failed getting the index, check get_index_vectorstore_wiki_nyc()"

test_get_index_vectorstore_wiki_nyc("bla")

#%%

def test_create_chroma_db():
    #requirement
    test_docs=test_split_data_to_docs()
    embed_model = test_get_embed_model()

    #test
    vectorstore = create_chroma_db(test_docs, embed_model)
    assert isinstance(vectorstore, Chroma) , "Failed creating the chroma db, check create_chroma_db()"

test_create_chroma_db()

#%%

def test_load_chroma_db():
    embed_model = test_get_embed_model()
    vectorstore = load_chroma_db(embed_model)
    assert isinstance(vectorstore, Chroma) , "Failed loading the chroma db, check load_chroma_db()"
    return vectorstore

vectorstore = test_load_chroma_db()

# %%

def test_similarity_search():
    embed_model = get_embed_model(embed_model_id)
    db = FAISS.load_local(DB_PATH, embed_model)
    query = QUERY

    similar_context = similarity_search_doc(db, query)
    assert isinstance(similar_context, list) , "Failed getting the similar context, check similarity_search()"

similar_context = test_similarity_search()
# %%
