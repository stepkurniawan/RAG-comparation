
#Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_API_ENDPOINT")
openai.api_version = "2023-07-01-preview"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

response = openai.ChatCompletion.create(
  engine="ChatGPT",
  messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},
                {"role":"user","content":"Say hi to me."}
  ],
  temperature=0.7,
  max_tokens=800,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)
  
print(response.choices[0].message.content)
  # %%
