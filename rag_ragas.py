from typing import Optional
from dotenv import load_dotenv
import os

from StipVectorStore import StipVectorStore
load_dotenv()
hf_token = os.getenv('HF_AUTH_TOKEN')


from datasets import load_dataset, Dataset

# from langchain_community.chat_models import AzureChatOpenAI
# from langchain_community.llms import AzureOpenAI
# from langchain_community.embeddings import AzureOpenAIEmbeddings

from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from ragas import evaluate
from ragas.metrics import ContextPrecision, ContextRecall

from rag_llms import get_gpt35_llm, load_llm_tokenizer_hf_with_model
from rag_chains import retrieval_qa_chain_from_local_db

from rag_embedding import get_embed_model, embedding_ids
import time
import transformers

from rag_load_data import load_sustainability_wiki_langchain_documents, load_from_webpage, load_qa_rag_dataset, load_50_qa_dataset
import openai

from rag_vectorstore import similarity_search_doc, multi_similarity_search_doc


import pandas as pd
from LogSetup import logger

RAGAS_METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
QA_FULL_DATASET = "data/collection_ground_truth_ragas_chatgpt4.json"
HF_HUB_QA_LLAMA2_13B = "stepkurniawan/qa-rag-llama2-13B-chat-hf"
HF_HUB_TEST = "stepkurniawan/test"


#######################################
transformers.logging.set_verbosity_info()



# LOAD DATASET #######################################################################################


def load_dataset_from_pandas(df):
    dataset = Dataset.from_pandas(df)
    print("success load dataset from pandas dataframe")
    return dataset


# deprecated 
def make_eval_chains():
    # make eval chains
    eval_chains = {
        m.name: RagasEvaluatorChain(metric=m) 
        for m in RAGAS_METRICS
    }
    return eval_chains

# deprecated 
# make a dataframe to store the scores, the columns are: [query, result, context, and metrics in the ragas_metrics
def evaluate_RAGAS(chain_response):
    eval_chains = make_eval_chains()

    # context list 
    page_contents_array = [doc.page_content for doc in chain_response['source_documents']]

    # make a dict to save all the scores from ragas
    ragas_result = {}
    
    # make a table to save each question, answer, and score
    eval_df = pd.DataFrame(columns=['query', 'result', 'context'] + [m.name for m in RAGAS_METRICS])

    for name, eval_chain in eval_chains.items():
        score_name = f"{name}_score"
        print(f"{name}: {eval_chain(chain_response)[score_name]}")
        # save the score to ragas_result dict
        ragas_result[name] = eval_chain(chain_response)[score_name]

    # save the result to eval_df
    eval_df.loc[0] = [chain_response['query'], chain_response['result'], page_contents_array] + [ragas_result[m.name] for m in RAGAS_METRICS]

    return eval_df
    
    
# %%
def prepare_qa_dataset_ragas(PATH=QA_FULL_DATASET):
    """
    # ref: https://colab.research.google.com/github/explodinggradients/ragas/blob/main/docs/quickstart.ipynb#scrollTo=22eb6f97
    the output must be a hugging face dataset, so that I can use it in 
    result = evaluate(
                    fiqa_eval["baseline"], ---> this is our dataset
                    metrics=[
                            context_precision,
                            faithfulness,
                            answer_relevancy,
                            context_recall,
                            harmfulness,
                        ],
                        )

    a complete dataset should look like this: 
    Dataset({
        features: ['question', 'ground_truths', 'answer', 'contexts'],
        num_rows: 30
    })
    - question: list[str] - These are the questions you RAG pipeline will be evaluated on.  --> from the user input
    - answer: list[str] - The answer generated from the RAG pipeline and give to the user.  --> from RAG (generator)
    - contexts: list[list[str]] - The contexts which where passed into the LLM to answer the question. --> from the RAG (retriever)
    - ground_truths: list[list[str]] - The ground truth answer to the questions. (only required if you are using context_recall) --> from the user input

    In our json, question is question, but the answer should be renamed to ground truth.

    Pseudocodes: 
    - We get the full dataframe from JSON ground_truth wiki that contains the contexts, question, answer, and ground_truths.
    - we remove the summary and contexts columns. 
    - convert the dataframe to hugging face dataset with train-test 80-20 split.
    - upload it to hugging face. 

    """
    
    df = pd.read_json(PATH)
    df = df.drop(columns=['summary', 'contexts']) 
    # the columns here should be just "question" and "ground_truths"
    # because the "contexts" and "answer" should come from RAG retriever and generator

    # create hugging face dataset from pandas dataframe
    dataset = Dataset.from_pandas(df)
    print("success creating dataset from pandas dataframe")

    # split the dataset to train-test 80-20 split
    sp_dataset = dataset.train_test_split(test_size=0.2, shuffle=True)

    return sp_dataset





# %% try to answer those 3 questions using RAG, and create a dataset 
"""
the dataset should look like: 
DatasetDict({
    baseline: train({
        features: ['question', 'ground_truths', 'answer', 'contexts'],
        num_rows: 3
    })
})
"""

#deprecated
def simplify2_source_document(docs):
    output = ""
    for index, doc in enumerate(docs):
        # this function should change Document to be just a string
        output = output + f"| title {index+1}: {doc.metadata['title']} \n| page_content {index+1}: {doc.page_content}\n\n"

    return output

def simplify_source_document(docs):
    """
    get docs, and make it list of list of strings
    """
    output = []
    for index, doc in enumerate(docs):
        # this functions with take the document's metadata's title and page_content, and put it into list of list of strings
        title = doc.metadata['title']
        page_content = doc.page_content
        output.append(f"| title {index+1}: {title} \n| page_content {index+1}: {page_content}\n\n")

    return output

def generate_contexts_answer(dataset:Dataset, llm, db):
    """
    input: Dataset with columns: question, ground_truths
    output: Dataset with columns: question, ground_truths, contexts, answer

    the context is coming from the retriever
    the answer is coming from the generator
    """

    try:
        print("to do modification on the dataset, change to dataframe")
        dataset = dataset.to_pandas()
    except:
        print("dataset is already a dict. just proceed.")

    # get the question from the dataset
    questions = dataset['question']

    # find the contexts using the retriever (similarity search)
    # the context is from rag_chains.retrieval_qa_chain_from_local_db()
    # but then it's stuffed ( summarized )
    # so we need to find the summarized context -> turns out we cant...
    # https://python.langchain.com/docs/use_cases/question_answering/how_to/vector_db_qa

    # we can run direclty the generator, and get both the answer (or result) and the contexts (or source_documents) after
    qa_chain = retrieval_qa_chain_from_local_db(llm=llm, vectorstore=db) 

    # create a series to store the contexts and answer
    rag_contexts = []
    rag_answer = []

    # loop the query, because each row have different question
    for index, query in enumerate(questions):
        qa_chain_result = qa_chain({'query' : query}) # result has 4 keys: questions,ground_truths, result, source_documents 
        answer = qa_chain_result['result']
        contexts = qa_chain_result['source_documents'] 
        print(f"{index}: question: ", query)
        print(f"{index}: answer: ", answer)
        # print("the contexts are: ", contexts)

        # insert the contexts to the series "contexts"
        rag_contexts.append(contexts)

        # insert the answer to the series "contexts"
        rag_answer.append(answer)

    # rag_contexts is a list of list of documents. 
    # it doesnt work. Lets change it to be a list of list of strings
    # no its still not working, you have to change it to a list of strings (per rows)
    rag_contexts = [simplify_source_document(docs) for docs in rag_contexts]
    
    # insert the contexts and answer to the dataset
    dataset['answer'] = rag_answer
    dataset['contexts'] = rag_contexts 

    # data processing: change ground_truth column from string to be a list[list[string]]
    dataset['ground_truths'] = [[x] for x in dataset['ground_truths']]
    
    # change to dataset again
    dataset = Dataset.from_dict(dataset)
    return dataset




def generate_answer_using_qa_chain(qa_dataset:Dataset, qa_chain, save_path:Optional[str]=None):
    for i in range(len(qa_dataset)):
        response = qa_chain({'query' : qa_dataset['question'][i]})
        response['result'] = response['result'].rstrip('\n') # clean data 
        # contexts_retrieved = [doc.page_content for doc in response['source_documents']]
        output_df = pd.concat([output_df, 
                               pd.DataFrame([{'query': response['query'], 
                                              'ground_truths': qa_dataset['ground_truths'][i] , 
                                              'answer': response['result'], 
                                              'contexts': response['source_documents']}])], ignore_index=True)
        
        if save_path is not None:
            # save output as a csv file and json
            output_df.to_csv(save_path + qa_chain.name + ".csv", index=False)
            output_df.to_json(save_path + qa_chain.name + ".json")
        
    return output_df




# %%




### load the llm, embed_model, and db

# llm = load_llm_tokenizer_hf_with_model(LLAMA2_7B_CHAT_MODEL_ID) 
# embed_model = get_embed_model(embedding_ids['BGE_LARGE_ID'])
# db = load_local_faiss_vector_database(embed_model)
# qa_dataset = load_50_qa_dataset()
# qa_dataset = qa_dataset['train'][:3] # if you limit using [:2] -> it will become a dictionary, not a dataset
# gen_dataset = pushing_dataset_to_hf_hub_pipeline(llm, db, qa_dataset, HF_HUB_PATH=HF_HUB_QA_LLAMA) 

# %% EVALUATE #################################################################################

def ragas_evaluate_push(dataset):
    """
    input: dataset from HF
    output: ragas evaluation result
    """
    # load the Dataset from DatasetDict
    try: 
        dataset = dataset['train']
    except:
        print("ragas_evaluate(): no train dataset found, using the dataset directly")

    # evaluate
    start_time = time.time() 

    result = evaluate(
                    dataset,
                    metrics=[
                            context_precision,
                            faithfulness,
                            answer_relevancy,
                            context_recall,
                        ],
                        )
    end_time = time.time() 
    execution_time = end_time - start_time  # calculate the execution time
    print(f"Execution time: {execution_time:.2f} seconds, or {execution_time/60:.2f} minutes, or {execution_time/3600:.2f} hours")
    
    result = result.to_pandas()

    result_dataset = load_dataset_from_pandas(result)

    subset_name = dataset.config_name # get the name of the llm from the dataset
    result_dataset.push_to_hub(HF_HUB_RAGAS, token=hf_token, config_name=subset_name)
    
    return result_dataset





# %% RETRIEVER EVALUATION ########################################################

def retriever_evaluation(
    dataset: Dataset,
    path_to_save: str = 'data/retriever_evaluation.csv',
    db: Optional[StipVectorStore] = None,
    top_k: Optional[int] = 3,
):
    """
    calls RAGAS to evaluate retriever
    Usage: evaluate_retriever(multisimilaritysearch(StipVectorStore("chroma").load_vectorstore("vectorstore_path")), qa_dataset, 6)
    Input: vectorstore_database that has column "contexts" 
    returns a dataframe with additional columns: context_precision, context_recall
    """

    ## do multisimilarity search, only when there is no "contexts" column
    if "contexts" not in dataset.column_names:
        print("dataset doesnt have 'contexts' column. Do multisimilarity search")
        try:
            contexted_dataset = multi_similarity_search_doc(db, dataset, top_k)
        except:
            raise Exception("Error: dataset doesnt have 'contexts' column, and failed to do multisimilarity search, check if you have the correct db and topk")
    elif isinstance(dataset, pd.DataFrame):     # if dataset is a dataframe, change to to Dataset
        print("dataset is a dataframe. Change to Dataset")
        contexted_dataset = Dataset.from_pandas(dataset)
    else:
        print("dataset already has 'contexts' column. Skip multisimilarity search")
        contexted_dataset = dataset
        
    context_precision = ContextPrecision()
    context_recall = ContextRecall(batch_size=10)
    start_time = time.time() 
    contexted_dataset = context_precision.score(contexted_dataset)
    contexted_dataset = context_recall.score(contexted_dataset)
    evaluated_df = pd.DataFrame(contexted_dataset)
    print("success evaluate retriever")
    end_time = time.time()
    total_time = end_time - start_time
    # if folder doesnt exist, create it. folder is without the .csv filename
    # folderpath = path_to_save, but without the filename
    folderpath = os.path.dirname(path_to_save)
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    
    evaluated_df.to_csv(path_to_save)
    print("success save retriever evaluation CSV to: ", path_to_save)
    print(f'Execution time: {total_time:.2f} seconds, or {total_time/60:.2f} minutes')
    logger.info("success save retriever evaluation CSV to: " + path_to_save)
    logger.info(f'Execution time: {total_time:.2f} seconds, or {total_time/60:.2f} minutes')
    
    return evaluated_df

def load_retriever_evaluation(path):
    """
    input: path to retriever evaluation csv
    output: dataframe
    """
    df = pd.read_csv(path)
    return df

# result = retriever_evaluation(qa_dataset['train'])
# #save result to csv under data dir
# result.to_csv('data/retriever_evaluation.csv')
# print(result)


#%% GENERATOR EVALUATION ########################################################
# from ragas.langchain.evalchain import RagasEvaluatorChain

# faithfulness_chain = RagasEvaluatorChain(metric=faithfulness)
# answer_rel_chain = RagasEvaluatorChain(metric=answer_relevancy)
# context_rel_chain = RagasEvaluatorChain(metric=context_precision)
# context_recall_chain = RagasEvaluatorChain(metric=context_recall)



def evaluate_qa_dataset_with_response(response, QUESTION_DATASET, qa_chain , FOLDER_PATH:Optional[str]=None):
    """
    Call ragas on a qa_chain, and evaluate the result using faithfulness, answer_relevancy, context_precision, context_recall
        based on a question dataset.
    This function combines both: generation and evaluation.
    input: response, QUESTION_DATASET
        - response is a dataframe that has query, result, ground_truths, and source_documents
    output: dataframe with columns: query, result, context, and metrics in the ragas_metrics

    """
    output_df = pd.DataFrame()
    for i in range(0, len(QUESTION_DATASET["question"])):
        ## EVALUATION using RAGAS
        faithfulness_eval = faithfulness_chain(response) # add 'faithfulness_score': 1.0 to the dict
        answer_rel_eval = answer_rel_chain(response) # add answer_relevancy_score': 0.991 to the dict
        context_rel_eval = context_rel_chain(response)# add context_precision_score': 0.8333 to the dict
        context_recall_eval = context_recall_chain(response) # context_recall_score': 1.0 to the dict

        output_df = pd.concat([output_df,
                                pd.DataFrame([{'query': response['query'],  # in ragas: question
                                                'ground_truths': response['ground_truths'] ,  # ground_truth
                                                'result': response['result'], # answer
                                                'source_documents': response['source_documents'],
                                                'faithfulness_score': faithfulness_eval['faithfulness_score'],
                                                'answer_relevancy_score': answer_rel_eval['answer_relevancy_score'],
                                                'context_precision_score': context_rel_eval['context_precision_score'],
                                                'context_recall_score': context_recall_eval['context_recall_score']}])], 
                                                ignore_index=True) # contexts
        
    # save output as a csv file and json
    if FOLDER_PATH is not None:
        output_df.to_csv(FOLDER_PATH + qa_chain.name + "_eval.csv", index=False) # only for excel
        output_df.to_json(FOLDER_PATH + qa_chain.name + "_eval.json")

    return output_df

