# !pip install transformers
# conda install -c pytorch -c nvidia faiss-gpu=1.7.4
# pip install datasets

from transformers import pipeline
# classifier = pipeline('sentiment-analysis') # default No model was supplied, defaulted to distilbert-base-uncased-finetuned-sst-2-english and revision af0f99b (https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english).
classifier = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")


print(classifier("I've been waiting for a HuggingFace course my whole life."))

results = classifier(["We are very happy to show you the 🤗 Transformers library.",
                        "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")


