"""
THIS FILE: is used to upload datasets to huggingface hub
the input of this file was JSON file that was created from Sustainability+Methods_dump.xml
the dump file was downloaded from wikiMedia
to be able to make it public easily, the author decided to create a dataframe and upload the dataset to huggingface hub
"""

#%% pip install ipywidgets

from datasets import load_dataset, Dataset
import os
import xmltodict
import json
import pandas as pd
import html
from huggingface_hub import notebook_login

JSON_PATH = "data/Sustainability+Methods_dump.xml.json"
COLLECTION_GROUND_TRUTH_RAGAS_CHATGPT4_JSON_PATH = "data/collection_ground_truth_ragas_chatgpt4.json"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SUST_WIKI_HUB_1 = "stepkurniawan/sustainability-methods-wiki"
SUST_WIKI_HUB_2 = "stepkurniawan/qa_sustainability_wiki"


#%% LOGIN
# notebook_login()
# or use huggingface-cli login in the terminal
#%%

def create_json_from_xml(xml_path):
    with open(xml_path, 'r', encoding='utf-8') as f:
        data_dict = xmltodict.parse(f.read())
        json_data = json.dumps(data_dict, indent=4)
        
    with open(xml_path+".json", "w") as json_file:
        json_file.write(json_data)
    
def load_dict_from_json(json_path):
    with open(json_path, 'r') as f:
        data_dict = json.load(f)
    return data_dict

def create_dataset_to_current_directory():
    """
    ref: https://huggingface.co/docs/datasets/v2.14.5/en/package_reference/main_classes#datasets.Dataset.from_dict
    """
    ds = load_dataset("rotten_tomatoes", split="validation")
    dir_path = os.path.join(CURRENT_DIR, "data")
    ds.info.write_to_directory(dir_path)
    return ds

def load_dataframe_from_json(json_path):
    dict = load_dict_from_json(json_path)
    pages = dict['mediawiki']['page']
    df = pd.DataFrame(pages)
    df['text'] = df['revision'].apply(lambda x: html.unescape(x['text']['#text']))
    df.drop(columns=['revision', 'ns', 'id'], inplace=True)

    print("success load dataframe from json: ", json_path)

    return df

def load_dataset_from_pandas(df):
    dataset = Dataset.from_pandas(df)
    print("success load dataset from pandas dataframe")
    return dataset

def create_csv_from_dataframe(df):
    df.to_csv("data/Sustainability+Methods_dump.csv", index=False, encoding='utf-8', sep=';')
    print("success create csv from dataframe")





############################################################################################################################
#%% PREPARING DATAFRAME TO UPLOAD TO HUGGINGFACE
sustainability_df = load_dataframe_from_json(JSON_PATH)
create_csv_from_dataframe(sustainability_df)
my_dataset = load_dataset_from_pandas(sustainability_df)
# my_dataset = my_dataset.train_test_split(test_size=0.2, shuffle=True)
my_dataset.push_to_hub(SUST_WIKI_HUB_1)

#%% UPLOAD TO HUGGINGFACE
test_load_dataset_from_hf = load_dataset(SUST_WIKI_HUB_1)
print(test_load_dataset_from_hf)

# %%
