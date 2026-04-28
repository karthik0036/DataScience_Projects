from openai import OpenAI
import os
api_key = os.getenv("NVIDIA_API_KEY")

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = api_key
)


completion = client.chat.completions.create(
  model="nvidia/nemotron-3-nano-30b-a3b",
  messages=[{"role":"user","content":"Provide me an essay about LLM"}],
  temperature=1,
  top_p=1,
  max_tokens=16384,
  extra_body={"reasoning_budget":16384,"chat_template_kwargs":{"enable_thinking":True}},
  stream=True
)

for chunk in completion:
  if not chunk.choices:
    continue
  reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
  if reasoning:
    print(reasoning, end="")
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")
  

