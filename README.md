# RAG-comparation
Evaluate and Compare different combination of Retriever and Generator of RAG models

# Naming Convention
- StipEmbedding, StipKnowledgeBase, and StipVectorStore are class files to handle the embedding, knowledge base, and vector store of the model. 

Experiments: 
- ExperimentKnowledgeBase: Experiment to evaluate the knowledge base of the model
    - test if the knowledge base is loaded correctly 
    - understand the metadata of the knowledge base
    - compare knowledge base FAISS vs Chroma, and evaluate its RAGAS metrics
- ExperimentEmbedding
- ExperimentVectorStore
- ExperimentSimilaritySearch
- ExperimentLLM
- ExperimentAll
- ExperimentAll-tableview

Deprecated:
- rag_embedding: code for embedding. It is replaced by SipEmbedding
- rag_vectorstore: code for vectorstore. It is replaced by SipVectorStore

# Preparations
1. run.py and run_multi.py are the files I use to create the vectorstore and fill it with vectors according to the embeddings and the index distances. run.py for single embedding and idnex distance, run_multi.py for multiple embeddings and index distances. 
2. 


# Experiments
1. 

