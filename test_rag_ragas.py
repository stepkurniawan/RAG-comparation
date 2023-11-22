from rag_ragas import azure_open_ai

from dotenv import load_dotenv
import os
load_dotenv()
from datasets import Dataset
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
test_dataset.save_to_disk("data/test_dataset")

# %%

azure_open_ai()