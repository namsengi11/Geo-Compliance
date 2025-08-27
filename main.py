import torch

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model = pipeline(
  'text-generation',
  model='facebook/bart-large-cnn',
  device=device,
  max_length=256,
  truncation=True
)

llm = HuggingFacePipeline(pipeline=model)

template = PromptTemplate.from_template("Evaluate if feature: {feature_name} - ({feature_description}) is compliant with the following regulation: {regulation_text}, and provide reasoning for your answer.")

chain = template | llm
feature_name = "Curfew login blocker with ASL and GH for Utah minors"
feature_description = "To comply with the Utah Social Media Regulation Act, we are implementing a curfew-based login restriction for users under 18. The system uses ASL to detect minor accounts and routes enforcement through GH to apply only within Utah boundaries. The feature activates during restricted night hours and logs activity using EchoTrace for auditability. This allows parental control to be enacted without user-facing alerts, operating in ShadowMode during initial rollout."
regulation_text = "Utah Social Media Regulation Act"

response = chain.invoke({"feature_name": feature_name, "feature_description": feature_description, "regulation_text": regulation_text})
print(response)