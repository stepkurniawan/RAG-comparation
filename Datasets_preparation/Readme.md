# Readme

## prepare_dataset.py 
THIS FILE: is used to create Question and Answer table for RAGAS that has the columns: context, question, answer, and summary. 
The purpose is to create an automated ground truth from the dump file of the wikipedia articles (from sustainability methods wiki).
At the end, it outputs JSON file that we can use for RAGAS evaluation. 

## upload_datasets_hf.py
THIS FILE: is used to upload datasets to huggingface hub
the input of this file was JSON file that was created from Sustainability+Methods_dump.xml
the dump file was downloaded from wikiMedia
to be able to make it public easily, the author decided to create a dataframe and upload the dataset to huggingface hub