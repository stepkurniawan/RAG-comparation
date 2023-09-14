from ragas.metrics import faithfulness, answer_relevancy, context_relevancy, context_recall
from ragas.langchain import RagasEvaluatorChain
import pandas as pd


def make_eval_chains(ragas_metrics):
    # make eval chains
    eval_chains = {
        m.name: RagasEvaluatorChain(metric=m) 
        for m in ragas_metrics
    }
    
    return eval_chains
    
# make a dataframe to store the scores, the columns are: [query, answer, context, and metrics in the ragas_metrics]

def evaluate_RAGAS(eval_chains, chain_response, df):
    for name, eval_chain in eval_chains.items():
        score_name = f"{name}_score"
        print(f"{score_name}: {eval_chain(chain_response)[score_name]}")

    # store the scores in the dataframe
    df.loc[len(df)] = [chain_response['query'], chain_response['answer'], chain_response['context']] + [eval_chain(chain_response)[score_name] for name, eval_chain in eval_chains.items()]

    
# %%
