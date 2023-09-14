from langchain import PromptTemplate


custom_prompt_template = """ Use the following pieces of information to answer user's question.
If you don't know the answer, just say you don't know. Don't make up information yourself.

Context: {context}
Question: {query}

Only returns helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(
        input_variables=["context","query"],
        template=custom_prompt_template)
    
    # print(f"your custom_prompt is : {prompt.format()}")
    return prompt

