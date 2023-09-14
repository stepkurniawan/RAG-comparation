from langchain import PromptTemplate
from langchain.prompts import ChatPromptTemplate



custom_prompt_template = """ Use the following pieces of information to answer user's question.
If you don't know the answer, just say you don't know. Don't make up information yourself.

Question: {query}

Only returns helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(
        input_variables=["query"],
        template=custom_prompt_template)
    
    # print(f"your custom_prompt is : {prompt.format()}")
    return prompt

def set_custom_prompt_new():
    prompt = PromptTemplate.from_template(custom_prompt_template)
    return prompt