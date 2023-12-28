"""
Ref: https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/split_by_token
Splitting text has many forms. The simplest one is by characters. 
But sometimes you want to split by words, sentences, paragraphs, or other tokens.
Normally, language models are trained using token as a splitter. 
Example of token splitters:
- tiktoken
- spaCy
- SentenceTransformers
- NLTK
- HuggingFace tokenizer
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import tiktoken
import time

# VARIABLES

chuck_sizes_list = [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600]

# using tiktoken as tokenizer 

enc = tiktoken.get_encoding("cl100k_base")

def length_function(text: str) -> int:
    return len(enc.encode(text))

# The default list of split characters is [\n\n, \n, " ", ""]
# Tries to split on them in order until the chunks are small enough
# Keep paragraphs, sentences, words together as long as possible
def split_data_to_docs(data, chunk_size = 200, chunk_overlap_scale = 0.1):
    """
    input versi 2: Dict{source: str, documents: list of Document objects}
    output : list of Document objects
    """
    ### check wether the data is actually already a list of Document objects
    ## or is it my dict version with additional source

    if type(data) == list:
        documents = data
        docs_source = None
    else:
        docs_source = data['source'] if 'source' in data.keys() else None
        documents = data['documents'] if 'documents' in data.keys() else data

    start_time = time.time() # start timer

    splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", " "],
                chunk_size=chunk_size, 
                chunk_overlap=chunk_size*chunk_overlap_scale,
                length_function=length_function,
                )

    all_splits = splitter.split_documents(documents)

    end_time = time.time() # end timer
    total_time = end_time - start_time
    print(f"Splitting Docs time: {total_time:.2f} seconds, or {total_time/60:.2f} minutes")
    
    dict_split = {
        "source": docs_source,
        "documents": all_splits
    }
    
    return dict_split

