import getpass
from dotenv import load_dotenv
import os

########## MY IMPORTS ##########################
load_dotenv()
from StipKnowledgeBase import StipKnowledgeBase
from rag_splitter import split_data_to_docs
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
from StipVectorStore import StipVectorStore 
from rag_embedding import get_embed_model, embedding_ids

###################################


# os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

# Uncomment the following line if you need to initialize FAISS with no AVX2 optimization
# os.environ['FAISS_NO_AVX2'] = '1'

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

loader = TextLoader("requirements.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

########### MY CODE ################
suswiki_kb = StipKnowledgeBase("stepkurniawan/sustainability-methods-wiki", None, "huggingface_cache/suswiki_hf")
dict_data = suswiki_kb.load_documents(10)
docs = split_data_to_docs(data=dict_data, chunk_size=200, chunk_overlap_scale=0.1)

embeddings = get_embed_model(embedding_ids['HF_BER_ID_2'])


faiss_vs = StipVectorStore("faiss")
# chroma_vs = VectorStore("chroma")
faiss_vs.create_vectorstore(docs, embeddings) 
####################################




db = FAISS.from_documents(docs['documents'], embeddings)

query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)

print(docs)
