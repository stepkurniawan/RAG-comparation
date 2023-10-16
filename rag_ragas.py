from typing import Optional

from dotenv import load_dotenv
import os
load_dotenv()

from datasets import load_dataset, Dataset

from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.langchain import RagasEvaluatorChain
from ragas import evaluate

from rag_llms import load_llm_gpt35, load_llm_tokenizer_hf_with_model
from rag_llms import LLAMA2_13B_CHAT_MODEL_ID, LLAMA2_7B_CHAT_MODEL_ID, LLAMA2_70B_CHAT_MODEL_ID

from rag_embedding import get_retriever_embeddings
from rag_vectorstore import load_local_faiss_vector_database
import time

import pandas as pd

RAGAS_METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
QA_FULL_DATASET = "data/collection_ground_truth_ragas_chatgpt4.json"
HF_HUB_QA_DATASET = "stepkurniawan/qa_sustainability_wiki"
HF_HUB_QA_LLAMA = "stepkurniawan/qa-rag-llama"
HF_HUB_QA_LLAMA2_13B = "stepkurniawan/qa-rag-llama2-13B-chat-hf"
HF_HUB_TEST = "stepkurniawan/test"

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

def push_to_hf_hub(dataset, HF_DATASET_HUB = HF_HUB_QA_DATASET):
    # push to HF
    dataset.push_to_hub(HF_DATASET_HUB)

#### UPLOAD qa dataset TO HF HUB stepkurniawan/qa_sustainability_wiki####
# sp_dataset = prepare_qa_dataset_ragas()
# push_to_hf_hub(sp_dataset, HF_HUB_QA_DATASET)




# %% take 3 examples from these questions 
def load_qa_rag_dataset():
    # load from HF
    dataset = load_dataset(HF_HUB_QA_DATASET)
    print("success loading question answer dataset from HF")
    print("the first question is: ", dataset['train']['question'][0])
    print("the first ground truth is: ", dataset['train']['ground_truths'][0])
    return dataset

# qa_dataset = load_qa_rag_dataset()
# take the first 3 rows from the "train" dataset
# qa_dataset = qa_dataset['train'][:3]






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

    # get the question from the dataset
    questions = dataset['question']

    # find the contexts using the retriever (similarity search)
    # the context is from rag_chains.retrieval_qa_chain_from_local_db()
    # but then it's stuffed ( summarized )
    # so we need to find the summarized context -> turns out we cant...
    # https://python.langchain.com/docs/use_cases/question_answering/how_to/vector_db_qa

    # we can run direclty the generator, and get both the answer (or result) and the contexts (or source_documents) after
    from rag_chains import retrieval_qa_chain_from_local_db, final_result

    qa_chain = retrieval_qa_chain_from_local_db(llm=llm, vectorstore=db) 

    # create a series to store the contexts and answer
    rag_contexts = []
    rag_answer = []

    # loop the query, because each row have different question
    for index, query in enumerate(questions):
        qa_chain_result = final_result(qa_chain, query)
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



# %%
def pushing_dataset_to_hf_hub_pipeline(llm, db, qa_dataset, HF_HUB_PATH, subset_name:Optional[str]=None):
    """
    input: qa_dataset from HF
    output: qa_dataset with columns: question, ground_truths, contexts, answer
    """

    start_time = time.time() 
    gen_dataset = generate_contexts_answer(qa_dataset, llm, db)
    end_time = time.time() 

    execution_time = end_time - start_time  # calculate the execution time
    print(f"Execution time: {execution_time:.2f} seconds, or {execution_time/60:.2f} minutes, or {execution_time/3600:.2f} hours")

    # # %% save the dataset to HF
    
    hf_token = os.getenv('HF_AUTH_TOKEN')
    if subset_name is not None:
        gen_dataset.push_to_hub(HF_HUB_PATH, config_name=subset_name ,token=hf_token) # config name will create a new subset
    else:
        subset_name = str(llm.pipeline.model.name_or_path)
        subset_name = subset_name.split("/")[-1]
        gen_dataset.push_to_hub(HF_HUB_PATH,config_name=subset_name, token=hf_token)
        gen_dataset.push_to_hub(HF_HUB_PATH, token=hf_token)
    print("success pushing the dataset to HF")

    return gen_dataset


### load the llm, embed_model, and db
# llm = load_llm_gpt35()


llm = load_llm_tokenizer_hf_with_model(LLAMA2_13B_CHAT_MODEL_ID) 
embed_model = get_retriever_embeddings()
db = load_local_faiss_vector_database(embed_model)
qa_dataset = load_qa_rag_dataset()
qa_dataset = qa_dataset['train'][:3] # limit only 100 rows max
gen_dataset = pushing_dataset_to_hf_hub_pipeline(llm, db, qa_dataset, HF_HUB_QA_LLAMA) # TODO: give subset_name

# %% EVALUATE #################################################################################

def ragas_evaluate(dataset):
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
                            # harmfulness,
                        ],
                        )
    end_time = time.time() 
    execution_time = end_time - start_time  # calculate the execution time
    print(f"Execution time: {execution_time:.2f} seconds, or {execution_time/60:.2f} minutes, or {execution_time/3600:.2f} hours")
    
    result = result.to_pandas()
    return result

fiqa_eval = load_dataset("explodinggradients/fiqa", "ragas_eval")['baseline']
qa_dataset = load_dataset(HF_HUB_QA_LLAMA, "Llama-2-13b-chat-hf")
result = ragas_evaluate(qa_dataset)