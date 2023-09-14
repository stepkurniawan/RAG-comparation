from langchain.llms import CTransformers # to use CPU only

def load_llm_ctra_llama27b():
    """
    loading CTransformers model
    """
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.0

    )
    return llm

