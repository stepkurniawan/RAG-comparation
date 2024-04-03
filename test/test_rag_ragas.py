from rag_ragas import azure_open_ai_old_version, activate_azure_ragas, get_azure_open_ai
from datasets import load_dataset, Dataset
from ragas import evaluate


from dotenv import load_dotenv
import os
load_dotenv()

from ragas.metrics import (
    context_precision,
    answer_relevancy,
    faithfulness,
    context_recall,
)

hf_token = os.getenv('HF_AUTH_TOKEN')
HF_HUB_TEST = "stepkurniawan/test"


# %% TESTING TOKEN AND HF

# create a test dataset
test_dataset = Dataset.from_dict(
    {
        "pokemon": ["bulbasaur", "squirtle", "charmander"], 
        "type": ["grass", "water", "fire"],
     })


test_dataset.push_to_hub(HF_HUB_TEST, config_name="starters" ,token = hf_token)
# create local backup
# test_dataset.save_to_disk("data/test_dataset")

# %% run azure open ai

# azure_open_ai_old_version() # check deployment
azure_client = get_azure_open_ai()
# activate_azure_ragas()


# %% TESTING RAGAS

# fiqa_eval = load_dataset("explodinggradients/fiqa", "ragas_eval")['baseline']
qa_dataset = load_dataset("stepkurniawan/qa-rag-llama", "Llama-2-13b-chat-hf", download_mode="force_redownload")
qa_dataset = qa_dataset['train']
result = evaluate(
                    qa_dataset,
                    metrics=[
                            context_precision,
                            faithfulness,
                            answer_relevancy,
                            context_recall,
                        ],
                        )

print(result)

# %%
