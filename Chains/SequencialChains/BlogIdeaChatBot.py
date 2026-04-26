from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
import os
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)

model=ChatHuggingFace(llm=llm)

parser=StrOutputParser()

idea_prompt=PromptTemplate(
    template="Generate an intresting blog idea about {topic}",
    input_variables=['topic']
)
outline_prompt=PromptTemplate(
    template="Create a detailed blog outline for this idea: {idea}",
    input_variables=["idea"]
)

title_prompt = PromptTemplate(
    
    template="Generate a catchy blog title from this outline: {outline}",
    input_variables=["outline"]
)

chain=idea_prompt|model|parser|outline_prompt|model|parser|title_prompt|model|parser

result=chain.invoke({"topic":"Unemployment in India"})





print(result)


