from datasets import load_dataset, Dataset

from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.langchain import RagasEvaluatorChain
from ragas import evaluate


import pandas as pd

RAGAS_METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
QA_FULL_DATASET = "data/collection_ground_truth_ragas_chatgpt4.json"
HF_HUB_QA_DATASET = "stepkurniawan/qa_sustainability_wiki"

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

qa_dataset = load_qa_rag_dataset()
# take the first 3 rows from the "train" dataset
qa_dataset = qa_dataset['train'][:3]





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
    for query in questions:
        qa_chain_result = final_result(qa_chain, query)
        answer = qa_chain_result['result']
        contexts = qa_chain_result['source_documents']
        print("the answer is: ", answer)
        print("the contexts are: ", contexts)

        # insert the contexts to the series "contexts"
        rag_contexts.append(contexts)

        # insert the answer to the series "contexts"
        rag_answer.append(answer)

    # insert the contexts and answer to the dataset
    dataset['contexts'] = rag_contexts
    dataset['answer'] = rag_answer

from rag_llms import load_llm_gpt35
from rag_embedding import get_retriever_embeddings
from rag_vectorstore import load_local_faiss_vector_database

llm = load_llm_gpt35()

embed_model = get_retriever_embeddings()

db = load_local_faiss_vector_database(embed_model)
generate_contexts_answer(qa_dataset, llm, db)



# %%