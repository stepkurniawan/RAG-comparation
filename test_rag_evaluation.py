from rag_ragas import evaluate_RAGAS


# %% TESTING RAGAS METRICS #####################################################

def test_evaluate_RAGAS():
    # must run the previous test functions first
    ragas_result = evaluate_RAGAS(test_chain_response)
    # assert isinstance(ragas_result, dict) , "Failed getting the ragas_result, check test_evaluate_RAGAS()"
    return ragas_result

ragas_rslt = test_evaluate_RAGAS()
print(ragas_rslt)

# %%