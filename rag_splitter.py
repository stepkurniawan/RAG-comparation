from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

# using tiktoken as tokenizer 

enc = tiktoken.get_encoding("cl100k_base")

def length_function(text: str) -> int:
    return len(enc.encode(text))

# The default list of split characters is [\n\n, \n, " ", ""]
# Tries to split on them in order until the chunks are small enough
# Keep paragraphs, sentences, words together as long as possible
def split_data_to_docs(data, chunk_size = 200):
    splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", " ", ""],
                chunk_size=chunk_size, 
                chunk_overlap=chunk_size*0.2,
                length_function=length_function,
                )

    all_splits = splitter.split_documents(data)

    return all_splits
