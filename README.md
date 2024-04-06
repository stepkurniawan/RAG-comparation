# RAG-comparation
Evaluate and Compare different combination of Retriever and Generator of RAG models. In this experiment we will compare RAG elements as seen in the picture: 
![image](https://github.com/stepkurniawan/RAG-comparation/assets/11275004/89aebb2f-74b5-4bd9-a1ca-a9303a64c9d0)
And then we will evaluate the result using RAGAS https://docs.ragas.io/en/latest/index.html
The main files here depends on which experiment. All of the experiments have "Experiment" as their starting name. 
The results of the experiments are stored in the `experiments` folder.
Here are some results: 
![image](https://github.com/stepkurniawan/RAG-comparation/assets/11275004/b470675c-eb1f-4bf4-9a73-e17b4092780f)
![image](https://github.com/stepkurniawan/RAG-comparation/assets/11275004/f3ced946-40fe-4d28-920a-c399274c4b89)




# Preparations
1. use ```pip install -r requirements.txt``` to install the required packages
2. run.py and run_multi.py are the files I use to create the vectorstore and fill it with vectors according to the embeddings and the index distances. run.py for single embedding and idnex distance, run_multi.py for multiple embeddings and index distances. (It will take a long time to run run_multi.py)

# Naming Convention
- StipEmbedding, StipKnowledgeBase, and StipVectorStore are class files to handle the embedding, knowledge base, and vector store of the model. 

# Experiments: 
- **ExperimentKnowledgeBase**: Experiment to evaluate the knowledge base of the model
    - test if the knowledge base is loaded correctly 
    - understand the metadata of the knowledge base
    - compare knowledge base FAISS vs Chroma, and evaluate its RAGAS metrics

- **ExperimentEmbedding**: Comparing embedding models: GTE, BGE, and UAE
    - test if the embedding is loaded correctly
    - create vectorstore for each embeddings
    - load vector store for each embeddings
    - retrieve using multi similarity search
    - evaluate the results of the retrieval using RAGAS metrics
    - compare embedding GTE, BGE, and UAE, and evaluate its RAGAS metrics by creaintg a dataframe table 
    - the CSVs are saved in the `experiments/Embeddings/` folder

- **ExperimentVectorStore**: comparing vectorstore models: FAISS, Chroma
    - load vector store for each vectorstore
    - split text into chunks
    - create vectorstore for FAISS and Chroma & save it to local
    - retrieve using multi similarity search
    - evaluate the results of the retrieval using RAGAS metrics
    - compare vectorstore FAISS, Chroma, and Sip, and evaluate its RAGAS metrics by creaintg a dataframe table 
    - create visualization out of the results
    - the CSVs are saved in the `experiments/vectorstore_comp/` folder

- **ExperimentSimilaritySearch**: comparing similarity search models: Eucledian, Cosine, and MIP
    - load vector store for each vectorstore
    - for each vectorstore
        - for each similarity search model
            - create vectorstore 
    - retrieve using multi similarity search
    - evaluate the results of the retrieval using RAGAS metrics
    - combine the results into 2 dataframes
    - the CSVs are saved in the `experiments/distance_metrics_comp/` folder

- **ExperimentLLM**: comparing LLM models: Llama2, Mistral, and GPT3.5
    - load LLM 
    - test each LLM to answer a question 
    - generate answer for 50 question using each LLM & save the result locally
    - data cleaning
    - evaluate the results using RAGAS metrics
    - the JSONS and CSVs are saved in the `experiments/Llm/` folder

- **ExperimentAll**: comparing each component of the RAG model
    - set up parameters
        - GENERATE_FLAG: for generating the answer - csv and json, use it if you trigger gpt3.5 (because its expensive)
        - EVALUATE_FLAG: to triggere ragas evaluation
    - load or otherwise create vector store if it wasnt existing
    - run_all: for all knowledge base, embedding, vector store, index distance, k, and LLM and store it locally
    - evaluate_all the results using RAGAS metrics and store it locally
    - the JSONS and CSVs are saved in the `experiments/ALL/` folder

- **ExperimentAll-tableview**: to further process the results of the ExperimentAll
    - load the results of the ExperimentAll
    - create a table view of the results
    - create visualizations

Deprecated: There are packages that are moved to the `deprecated` folder.
    Note: some experiments are still using the deprecated packages.
- rag_embedding: code for embedding. It is replaced by SipEmbedding
- rag_vectorstore: code for vectorstore. It is replaced by SipVectorStore


# Folders: 
- Deprecated: to store deprecated files
- tests: to store test files
- vectorstores: to store large vectorstore files (FAISS, Chroma)
    - vectorstore/db_chroma: to store Chroma vectorstore
    - vectorstore/db_faiss: to store FAISS vectorstore
- Documentations: to store documentation files for my papers, such as visualizations
- data: to store data files before I created `experiments` folder. important ones are:
    - hand_picked_questions_3.json : handpicked questions for SustainabilityQA
    - Sustainability+Methods_dump_checklist.csv : checklist to generate question answer pairs from which page of Sustainability Wiki
    - Sustainability+Methods_dump.csv
    - Sustainability+Methods_dump.xml.json
    - wiki_checklist_segmented_new.csv
    - and folders which contains the results of chunk splitting strategy (split size and overlap)
- Datasets_preparation: to converts unstructured Suswiki xml dump to structured csv files with questions and answers, and upload it to hugging face
- RAGAS notebook: just as a way for me to store RAGAS reference materials. 
- huggingface_cache: to store datasets that has been downloaded from Hugging Face

#### Logfile
The logfile contains info such as how long does the experiment run. 

#### List of Ablation todo.xlsx
Just a file where I track which combination should I prioritize when running the ExperimentAll
