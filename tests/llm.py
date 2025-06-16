from src.core.llm import Agent

llm=Agent.from_TaiwanAIRAP("Llama-3.1-8B-Instruct")
#llm=Agent.from_openai("gpt-4o")
generation = llm.generate("說一句話稱讚你的貓")
print(generation)

