import os
import time
import pandas as pd
from rag_vectorstore import similarity_search_doc, multi_similarity_search_doc
from LogSetup import logger
from rag_ragas import retriever_evaluation
from StipVectorStore import StipVectorStore
from StipKnowledgeBase import load_suswiki, load_wikipedia, load_50_qa_dataset
from StipEmbedding import StipEmbedding
import multiprocessing

####
# EMBED_MODELS = [ StipEmbedding("uae").embed_model, StipEmbedding("gte").embed_model, StipEmbedding("bge").embed_model]
EMBED_MODELS = [ StipEmbedding("uae").embed_model, StipEmbedding("gte").embed_model]
CHUNK_SIZE = 200
CHUNK_OVERLAP_SCALE = 0.1
TOP_KS = [1,2,3]
# INDEX_DISTANCES = ["cosine", "ip"]
INDEX_DISTANCES = ["cosine"]
# VECTORSTORES = ( StipVectorStore("chroma"), StipVectorStore("faiss"))
VECTORSTORES = ( StipVectorStore("faiss"),)
QUESTION_DATASET = load_50_qa_dataset()['train']

#### Load Knowledge Bases
suswiki_kb = load_suswiki()
wikipedia_kb = load_wikipedia()

# #%%### retriever
# ### create ALL wikipedia vector stores
# # Get the total number of iterations
# total_iterations = len(VECTORSTORES) * len(EMBED_MODELS) * len(INDEX_DISTANCES)
# iteration = 0

# for vector_str in VECTORSTORES:
#     print(f"Processing VECTORSTORE: {vector_str.vectorstore_name}")
    
#     for embed_model in EMBED_MODELS:
#         print(f"  Using EMBED_MODEL: {embed_model.model_name}")
        
#         for index_dist in INDEX_DISTANCES:
#             iteration += 1
#             print(f"    Applying INDEX_DISTANCE: {index_dist}")
#             print(f"    Starting iteration {iteration} of {total_iterations}...")
            
#             print("    start create vectorstore" + vector_str.vectorstore_name + "for wikipedia with embed model " + embed_model.model_name   + " and index distance " + index_dist)
#             wikipedia_vectorstore_faiss = vector_str.create_vectorstore(wikipedia_kb, embed_model, CHUNK_SIZE, CHUNK_OVERLAP_SCALE, index_dist)
            
#             print("    success create vectorstore for wikipedia with embed model " + embed_model.model_name  + " and index distance " + index_dist)
#             logger.info("    success create vectorstore for wikipedia with embed model " + embed_model.model_name  + " and index distance " + index_dist)
            
#             print(f"    Finished iteration {iteration} of {total_iterations}.")
            
#         print(f"  Finished processing EMBED_MODEL: {embed_model.model_name}")
        
#     print(f"Finished processing VECTORSTORE: {vector_str.vectorstore_name}")


#%% use multi processing for the vectorstore creation
from multiprocessing import Pool
from functools import partial
vector_str = StipVectorStore("faiss")
def multi_create_vectorstore(vector_str, kb, embed_model, chunk_size, chunk_overlap_scale, index_dist):
    print("    start create vectorstore" + vector_str.vectorstore_name + " for wikipedia with embed model " + embed_model.model_name   + " and index distance " + index_dist)
    wikipedia_vectorstore = vector_str.create_vectorstore(kb, embed_model, chunk_size, chunk_overlap_scale, index_dist)
    print("    success create vectorstore for wikipedia with embed model " + embed_model.model_name  + " and index distance " + index_dist)
    logger.info("    success create vectorstore for wikipedia with embed model " + embed_model.model_name  + " and index distance " + index_dist)
    return wikipedia_vectorstore

def multi_create_vectorstore_wrapper(args):
    return multi_create_vectorstore(*args)

def multi_create_vectorstore_pipeline(vector_str, kb, embed_models, chunk_size, chunk_overlap_scale, index_dists):
    # Get the total number of iterations
    # total_iterations = len(vector_str) * len(embed_models) * len(index_dists)
    iteration = 0
    
    # create a list of args for the multi processing
    args = []
    for vector in vector_str:
        for embed_model in embed_models:
            for index_dist in index_dists:
                args.append((vector, kb, embed_model, chunk_size, chunk_overlap_scale, index_dist))
    
    # create the pool
    with multiprocessing.Pool(processes=16) as pool:
        print("get start method: ", multiprocessing.get_start_method())
        results = pool.map(multi_create_vectorstore_wrapper, args)
    pool.close()
    pool.join()
    
    return results

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    test_pipeline = multi_create_vectorstore_pipeline(VECTORSTORES, wikipedia_kb, EMBED_MODELS, CHUNK_SIZE, CHUNK_OVERLAP_SCALE, INDEX_DISTANCES)

#%% 

# faiss_vs = StipVectorStore("faiss")
# chroma_vs = StipVectorStore("chroma")

# bge_embedding = StipEmbedding("bge").embed_model
# uae_embedding = StipEmbedding("uae").embed_model
# gte_embedding = StipEmbedding("gte").embed_model

# index_dist1 = "l2"
# index_dist2 = "cosine"
# index_dist3 = "ip"

# vector_str = faiss_vs.create_vectorstore(wikipedia_kb, bge_embedding, CHUNK_SIZE, CHUNK_OVERLAP_SCALE, index_dist1)

