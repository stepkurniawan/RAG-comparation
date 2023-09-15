#%% Imports

from rag_embedding import get_embed_model
from rag_vectorstore import get_index_vectorstore_wiki_nyc, similarity_search



from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS




embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
# %% 
def test_get_embed_model():
    
    embed_model = get_embed_model(embed_model_id)
    assert isinstance(embed_model, HuggingFaceEmbeddings) , "Failed getting the embedding model, check get_embed_model()"
    assert embed_model.model_name == embed_model_id
    return embed_model


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
# %% VECTOR STORE


def test_get_index_vectorstore_wiki_nyc(embed_model):
    embed_model = get_embed_model(embed_model_id)
    index = get_index_vectorstore_wiki_nyc(embed_model)
    # assert isinstance(index, VectorstoreIndexWrapper) , "Failed getting the index, check get_index_vectorstore_wiki_nyc()"
    assert isinstance(index.vectorstore, FAISS) , "Failed getting the index, check get_index_vectorstore_wiki_nyc()"

test_get_index_vectorstore_wiki_nyc("bla")
# %%

def test_similarity_search():
    similar_context = similarity_search()
    assert isinstance(similar_context, list) , "Failed getting the similar context, check similarity_search()"
