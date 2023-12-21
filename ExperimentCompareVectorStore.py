#%% 0. Import libraries
# !pip install bs4 chromadb tiktoken faiss-cpu accelerate xformers ragas

import os
from rag_embedding import get_embed_model
from rag_embedding import embedding_ids
from rag_vectorstore import VectorStore
from rag_vectorstore import get_index_vectorstore_wiki_nyc
from rag_vectorstore import svm_similarity_search_doc, similarity_search_doc, create_chroma_db, load_chroma_db
from rag_llms import load_llm_ctra_llama27b, load_llm_gpt35, load_llm_tokenizer_hf_with_model
from rag_llms import LLAMA2_13B_CHAT_MODEL_ID
# from rag_prompting import set_custom_prompt, set_custom_prompt_new, get_formatted_prompt
from rag_chains import retrieval_qa_chain_from_local_db, chain_with_docs, final_result
from rag_ragas import make_eval_chains, evaluate_RAGAS
from rag_splitter import split_data_to_docs
from KnowledgeBase import KnowledgeBase, HuggingFaceDataset

from langchain.llms import CTransformers # to use CPU only

from ragas.langchain import RagasEvaluatorChain

import pandas as pd


#%% 1. UI ##############################################################

#%% 2. Set up environment ############################################
QUERY = "When and where did the quantitative Content Analysis method originate?"
DB_PATH = "vectorstores/db_faiss"
LINK = "https://sustainabilitymethods.org/index.php/A_matter_of_probability"

#%% 3. RETRIEVER #####################################################

# 3.1 Load knowledge base / dataset #######
# data = load_from_webpage("https://sustainabilitymethods.org/index.php/A_matter_of_probability")
suswiki_kb = KnowledgeBase("stepkurniawan/sustainability-methods-wiki", None, "huggingface_cache/suswiki_hf")
# wikipedia_kb = KnowledgeBase("wikipedia", "20220301.simple", "huggingface_cache/wikipedia_hf")
data = suswiki_kb.load_documents()
print(f'the data is: {data}, it has {len(data)} documents')

# 3.1.1 limit data for test
# data = data[:10]

# 3.2 Split text into chunks ##############
docs = split_data_to_docs(data=data, chunk_size=200, chunk_overlap_scale=0.1)

# 3.3 embedding ###########
# embed_model = get_retriever_embeddings()
embed_model = get_embed_model(embedding_ids['HF_BER_ID_2'])

# 4 VECTOR STORE ######################################################
## create LOCAL FAISS
#%% # if folder DB_FAISS_PATH is empty, then run 

# 4.1 create vectorstore
faiss_vs = VectorStore("faiss")
chroma_vs = VectorStore("chroma")
# faiss_vs.create_vectorstore(docs, embed_model) 
chroma_db = create_chroma_db(docs, embed_model)

chroma_vs.create_vectorstore(docs, embed_model) 

## 4.5 Similiarity search
# faiss_db = faiss_vs.load_vectorstore()
chroma_db = chroma_vs.load_vectorstore()

# similar_response = similarity_search_doc(db, QUERY)
similar_docs = svm_similarity_search_doc(docs, QUERY, embed_model, 1)


###################### 4.5 EVALUATE RETRIEVER : context precision , recall, and F-measure
curated_qa_dataset = HuggingFaceDataset("stepkurniawan/qa-rag-llama", "50_QA", ["question", "ground_truths"])
retriever_evaluation = evaluate(
                    dataset,
                    metrics=[
                            context_precision,
                            faithfulness,
                            answer_relevancy,
                            context_recall,
                            # harmfulness,
                        ],
                        )

