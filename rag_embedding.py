from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings

embedding_ids = {
    'OPENAI_EMBEDDING_ID': 'text-embedding-ada-002',
    'BGE_LARGE_ID': "BAAI/bge-large-en-v1.5",
    'GTE_ID': "thenlper/gte-large",
    'HF_BERT_ID': "sentence-transformers/all-MiniLM-L12-v2", # https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2
    'HF_BER_ID_2': "sentence-transformers/all-MiniLM-L6-v2" 
}

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

embed_model_id_retriever = 'sentence-transformers/all-MiniLM-L6-v2'
embed_model_id_generator = 'sentence-transformers/all-MiniLM-L6-v2'


def get_embed_model(embed_model_id):
    if embed_model_id == embedding_ids['OPENAI_EMBEDDING_ID']:
        embed_model = OpenAIEmbeddings(model=embed_model_id)
        print(f'success load embed_model: {embed_model}')
        model_name = embed_model.model
        
    else:
        embed_model = HuggingFaceEmbeddings(
            model_name=embed_model_id,
            model_kwargs={'device': device},
            encode_kwargs={'device': device, 'batch_size': 32}
        )
        model_name = embed_model.model_name.split('/')[-1]

    return embed_model, model_name

def get_retriever_embeddings():
    embed_model = get_embed_model(embed_model_id_retriever)
    print(f'success load embed_model: {embed_model}')
    return embed_model


def get_generator_embeddings():
    embed_model = get_embed_model(embed_model_id_generator)
    print(f'success load embed_model: {embed_model}')
    return embed_model
