import time
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_core.vectorstores import VectorStore
from datasets import load_dataset, Dataset
from LogSetup import logger


import pandas as pd

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

VST = TypeVar("VST", bound="VectorStore")



def retrieval_qa_chain_from_local_db(llm, 
                                     vectorstore : Type[VST], 
                                     template_prompt = None, 
                                     k:int = 3):
    # Ref: https://python.langchain.com/docs/use_cases/question_answering/how_to/vector_db_qa
    # qa_chain_prompt  = PromptTemplate.from_template(template_prompt)

    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = 'stuff', # you can also change this to map reduce
        retriever = vectorstore.as_retriever(
            search_kwargs = {'k':k,
                             'score_threshold': 0.8}
            ),
        return_source_documents = True,        # retriever will ensure that llm will retrieve the information from the document that we have
        # chain_type_kwargs = {"prompt": qa_chain_prompt} 
    )
    vectorstore_name = str.split(str(vectorstore.__class__),".")[-1]
    alphabet_only = ''.join([char for char in vectorstore_name if char.isalpha()])
    try:
        qa_chain.name = llm.name + "_" + alphabet_only
    except:
        try: 
            qa_chain.name = llm.name 
        except:
            print("qa_chain.name not set")
    return qa_chain


#QA Model Function
def qa_bot( vectorstore, llm, qa_prompt):
    qa_bot = retrieval_qa_chain_from_local_db(llm, qa_prompt, vectorstore)

    return qa_bot

def qa_with_sources(llm, prompt_template, db):  
    chain_type_kwargs = {"prompt": prompt_template}
  
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
    )
    return qa_chain

# def final_result(qa_chain, query):
#     response = qa_chain({'query': query})
#     return response

#############
# LOAD QA CHAINS

def chain_with_docs(llm, unique_docs, question):
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain({"input_documents": unique_docs, "question": question}
                    #  ,return_only_outputs=True
                     )
    return response

#############
# GENERATE ANSWER FROM QA CHAIN
# DEPRECATED
# def generate_answer_from_qa_chain(qa_chain, qa_dataset, save_path:Optional[str]=None):
#     """
#     input: qa_chain, qa_dataset
#     output: response dataframe with columns: query, ground_truths, result, source_documents
#     """
#     print(f"Generating answer from QA chain: {qa_chain.name}")
#     response_out_df = pd.DataFrame()
#     for i in range(0, len(qa_dataset["question"])):
#         response = qa_chain({'query' : qa_dataset['question'][i]})
#         response['result'] = response['result'].rstrip('\n') # clean data 
#         response['ground_truths'] = qa_dataset['ground_truths'][i]

#         response_out_df = pd.concat([response_out_df,
#                                 pd.DataFrame([{'query': response['query'],  # in ragas: question
#                                                 'ground_truths': response['ground_truths'] ,  # ground_truth
#                                                 'result': response['result'], # answer
#                                                 'source_documents': response['source_documents'],}])],
#                                                 ignore_index=True) # contexts
        
#     # save output as a csv file and json
#     if save_path is not None:
#         response_out_df.to_csv(save_path + qa_chain.name + "_gen.csv", index=False) # only for excel
#         response_out_df.to_json(save_path + qa_chain.name + "_gen.json")
#         print(f"output created in path: {save_path}, check for CSV and JSON {qa_chain.name}")

#     return response_out_df

def generate_context_answer_langchain(qa_dataset:Dataset, llm, db: Type[VST], k, save_path:Optional[str]=None):
    """
    input: Dataset with columns: question, ground_truths
    output: Dataset with columns: question, ground_truths, contexts, answer

    the context is coming from the retriever
    the answer is coming from the generator
    """
    vectorstore_name = str(type(db)).split("'")[1].split(".")[-1] # FAISS or something
    print(f"Generating answer from QA Dataset: using {llm.name} and vector store: {vectorstore_name}...")
    start_time = time.time()
    response_out_df = pd.DataFrame()
    # get the question from the dataset
    questions = qa_dataset['question']
    qa_chain = retrieval_qa_chain_from_local_db(llm=llm, vectorstore=db, k=k) 

    for i in range(0, len(qa_dataset["question"])):
        response = qa_chain({'query' : qa_dataset['question'][i]})
        response['result'] = response['result'].rstrip('\n') # clean data 
        response['ground_truths'] = qa_dataset['ground_truths'][i]

        response_out_df = pd.concat([response_out_df,
                                pd.DataFrame([{'query': response['query'],  # in ragas: question
                                                'ground_truths': response['ground_truths'] ,  # ground_truth
                                                'result': response['result'], # answer
                                                'source_documents': response['source_documents'],}])],
                                                ignore_index=True) # contexts

    ## save the dataset
    end_time = time.time()

    if save_path is not None:
        response_out_df.to_csv(save_path + qa_chain.name + "_gen.csv", index=False)
        response_out_df.to_json(save_path + qa_chain.name + "_gen.json")
        print(f"output created in path: {save_path}, check for CSV and JSON {qa_chain.name}. Time taken: {(end_time-start_time)/60} minutes")
        logger.info(f"output created in path: {save_path}, check for CSV and JSON {qa_chain.name}")
    
    return response_out_df