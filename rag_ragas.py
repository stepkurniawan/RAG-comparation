from ragas.metrics import faithfulness, answer_relevancy, context_relevancy, context_recall
from ragas.langchain import RagasEvaluatorChain
import pandas as pd

RAGAS_METRICS = [faithfulness, answer_relevancy, context_relevancy]


def make_eval_chains():
    # make eval chains
    eval_chains = {
        m.name: RagasEvaluatorChain(metric=m) 
        for m in RAGAS_METRICS
    }
    return eval_chains
    
# make a dataframe to store the scores, the columns are: [query, result, context, and metrics in the ragas_metrics]

def evaluate_RAGAS(chain_response):
    eval_chains = make_eval_chains()
    
    # make a table to save each question, answer, and score
    eval_df = pd.DataFrame(columns=['query', 'result', 'context'] + [m.name for m in RAGAS_METRICS])

    for name, eval_chain in eval_chains.items():
        score_name = f"{name}_score"
        print(f"{score_name}: {eval_chain(chain_response)[score_name]}")
        # save the score to the dataframe
        eval_df[score_name] = eval_chain(chain_response)[score_name]

    return eval_df
    
# %%
