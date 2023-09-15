from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain

def retrieval_qa_chain_from_local_db(llm, prompt_template, db):
    # chain_type_kwargs = {"prompt": prompt_template}

    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = 'stuff', # you can also change this to map reduce
        retriever = db.as_retriever(search_kwargs = {'k':3}),
        return_source_documents = True,        # retriever will ensure that llm will retrieve the information from the document that we have
        # chain_type_kwargs = chain_type_kwargs
    )
    return qa_chain


#QA Model Function
def qa_bot( db, llm, qa_prompt):
    qa_bot = retrieval_qa_chain_from_local_db(llm, qa_prompt, db)

    return qa_bot

def qa_with_sources(llm, prompt_template, db):  
    chain_type_kwargs = {"prompt": prompt_template}
  
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
        chain_type_kwargs=chain_type_kwargs
    )
    return qa_chain

def final_result(query, qa_chain):
    response = qa_chain({'query': query})
    return response