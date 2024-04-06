from torch import cuda
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings

embedding_ids = {
    'openai': 'text-embedding-ada-002',
    'bge': "BAAI/bge-large-en-v1.5",
    'gte': "thenlper/gte-large",
    'hf_1': "sentence-transformers/all-MiniLM-L12-v2", # https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2
    'hf_2': "sentence-transformers/all-MiniLM-L6-v2", 
    'e5_mi': "intfloat/e5-mistral-7b-instruct", # too large
    'uae': "WhereIsAI/UAE-Large-V1"
}

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
class StipEmbedding: 
    def __init__(self, 
                 my_model_id : str
                 ):
        self.my_model_id = my_model_id
        if my_model_id not in embedding_ids.keys():
            raise ValueError(f"my_model_id must be one of {embedding_ids.keys()}")
        self.model_id = embedding_ids[my_model_id]
        self.model_name = None
        self.embed_model = None

        # Load embedding_model    
        if self.my_model_id == embedding_ids['openai']:
            self.embed_model = OpenAIEmbeddings(model=self.model_id)
            print(f'success load embed_model: {self.embed_model}')
            self.model_name = self.embed_model.model
        else:
            self.embed_model = HuggingFaceEmbeddings(
                model_name=self.model_id,
                model_kwargs={'device': device},
                encode_kwargs={'device': device, 'batch_size': 32}
            )
            self.model_name = self.embed_model.model_name.split('/')[-1]

    def get_model(self):
        return self.embed_model
    
