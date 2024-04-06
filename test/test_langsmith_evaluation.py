# https://docs.smith.langchain.com/evaluation/quickstart?ref=blog.langchain.dev

#%% CREATE A DATASET
from langsmith import Client

example_inputs = [
  "a rap battle between Atticus Finch and Cicero",
  "a rap battle between Barbie and Oppenheimer",
  "a Pythonic rap battle between two swallows: one European and one African",
  "a rap battle between Aubrey Plaza and Stephen Colbert",
]

client = Client()
dataset_name = "Rap Battle Dataset"

# Storing inputs in a dataset lets us
# run chains and LLMs over a shared set of examples.
dataset = client.create_dataset(
    dataset_name=dataset_name, description="Rap battle prompts.",
)
for input_prompt in example_inputs:
    # Each example must be unique and have inputs defined.
    # Outputs are optional
    client.create_example(
        inputs={"question": input_prompt},
        outputs=None,
        dataset_id=dataset.id,
    )

# %% DEFINE LLM

from langchain_community.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0)

# %% EVALUATE LLM

from langchain_community.smith import RunEvalConfig, run_on_dataset

eval_config = RunEvalConfig(
  evaluators=[
    # You can specify an evaluator by name/enum.
    # In this case, the default criterion is "helpfulness"
    "criteria",
    # Or you can configure the evaluator
    RunEvalConfig.Criteria("harmfulness"),
    RunEvalConfig.Criteria("misogyny"),
    RunEvalConfig.Criteria(
      {"cliche": "Are the lyrics cliche? "
      "Respond Y if they are, N if they're entirely unique."}
      )
  ]
)
run_on_dataset(
    client=client,
    dataset_name=dataset_name,
    llm_or_chain_factory=llm,
    evaluation=eval_config,
    verbose=True,
)


# %%
