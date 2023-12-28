#%% 0. Import libraries
# !pip install bs4 chromadb tiktoken faiss-cpu accelerate xformers ragas

import os

from rag_vectorstore import similarity_search_doc, multi_similarity_search_doc
# from rag_prompting import set_custom_prompt, set_custom_prompt_new, get_formatted_prompt

from rag_splitter import split_data_to_docs
from LogSetup import logger
from rag_ragas import retriever_evaluation

from StipVectorStore import StipVectorStore
from StipKnowledgeBase import StipKnowledgeBase, StipHuggingFaceDataset
from StipEmbedding import StipEmbedding

from langchain.llms import CTransformers # to use CPU only

from ragas.langchain import RagasEvaluatorChain
from ragas import evaluate
from ragas.metrics import ContextPrecision, ContextRecall


import pandas as pd


#%% 1. UI ##############################################################

#%% 2. Set up environment ############################################
QUERY = "When and where did the quantitative Content Analysis method originate?"
DB_PATH = "vectorstores/db_faiss"
LINK = "https://sustainabilitymethods.org/index.php/A_matter_of_probability"
OUTPUT_PATH = "experiments/vectorstore_comp"

#%% 3. RETRIEVER #####################################################

# 3.1 Load knowledge base / dataset #######
# data = load_from_webpage("https://sustainabilitymethods.org/index.php/A_matter_of_probability")
suswiki_kb = StipKnowledgeBase("stepkurniawan/sustainability-methods-wiki", None, "huggingface_cache/suswiki_hf")
# wikipedia_kb = KnowledgeBase("wikipedia", "20220301.simple", "huggingface_cache/wikipedia_hf")
dict_data = suswiki_kb.load_documents()
print(f'successful load data from {dict_data["source"]}')

# 3.1.1 limit data for test
# data = data[:10]

# 3.2 Split text into chunks ##############
docs = split_data_to_docs(data=dict_data, chunk_size=200, chunk_overlap_scale=0.1)

# 3.3 embedding ###########
# embed_model = get_retriever_embeddings()
embed_model = StipEmbedding("bge").embed_model
# embed_model = get_embed_model(embedding_ids['HF_BER_ID_2'])

#%% 4 VECTOR STORE ######################################################
## create LOCAL FAISS
def prepare_vectorstore():
    # 4.1 create vectorstore
    faiss_vs = StipVectorStore("faiss")
    chroma_vs = StipVectorStore("chroma")

    # Create vectorstore
    faiss_sus_bge = faiss_vs.create_vectorstore(docs, embed_model) 
    chroma_sus_bge = chroma_vs.create_vectorstore(docs, embed_model) 

    return faiss_sus_bge, chroma_sus_bge, faiss_vs, chroma_vs

# uncomment this to create vectorstore
# faiss_sus_bge, chroma_sus_bge, faiss_vs, chroma_vs = prepare_vectorstore()


#%% # 4.5 Similiarity search
## uncomment this to do similarity search
# database = chroma_sus_bge
# database_obj = chroma_vs
# similar_docs = similarity_search_doc(database, QUERY, 1)

###################### 4.5 EVALUATE RETRIEVER : context precision , recall, and F-measure

curated_qa_dataset = StipHuggingFaceDataset("stepkurniawan/sustainability-methods-wiki", "50_QA", ["question", "ground_truths"]).load_dataset()
dataset = curated_qa_dataset['train']

# 4.5.1 answer using similarity search
# create a table with question, ground_truths, and context (retrieved_answer)

def evaluate_retriever(vectorstore_database, qa_dataset, k=1):
    # 4.5.1 answer using similarity search
    # create a table with question, ground_truths, and context (retrieved_answer)
    contexted_dataset = multi_similarity_search_doc(vectorstore_database, qa_dataset, k)

    # answer using ragas evaluate
    context_precision = ContextPrecision()
    context_recall = ContextRecall(batch_size=10)
    contexted_dataset = context_precision.score(contexted_dataset)
    contexted_dataset = context_recall.score(contexted_dataset)

    contexted_df = pd.DataFrame(contexted_dataset)
    return contexted_df

def save_locally(contexted_df):
    # Check if the directory exists, if not, create it
    file_path = f"{OUTPUT_PATH}/{database_obj.vectorstore_name}_{database_obj.docs_source}_{database_obj.model_name}_{database_obj.chunk_size}_{database_obj.chunk_overlap_scale}"
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    contexted_df.to_csv(file_path +".csv", sep="|", )
    #save answer to json
    contexted_df.to_json(file_path+".json")


# contexted_df = evaluate_retriever(database, dataset, 1)
# save_locally(contexted_df)
    
# %%
import pandas as pd 

def read_local_results():
    output_df = pd.DataFrame() # this will countain columns: question, ground_truths, contexts, context_precision_FAISS, context_recall_FAISS, context_precision_Chroma, context_recall_Chroma
    vectorstores = ["faiss", "chroma"]
    # get the columns: question, ground_truths, contexts, context_precision_FAISS, context_recall_FAISS from each vectorstore
    for vectorstore in vectorstores:
        # load from local
        file_path = f"{OUTPUT_PATH}/{vectorstore}_sustainability-methods-wiki_bge-large-en-v1.5_200_0.1"
        df = pd.read_csv(file_path +".csv", sep="|", )
        df = df.drop(columns=['Unnamed: 0'])
        output_df = pd.concat([output_df, df], axis=1)

    return output_df
        

output_df = read_local_results()
print(output_df)


# %%
