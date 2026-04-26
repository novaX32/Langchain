from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task='text-generation',
    huggingfacehub_api_token=os.getenv("HF_TOKEN") 
)

model=ChatHuggingFace(llm=llm)

prompt=PromptTemplate(
    template="Explain {topic} in simple terms",
    input_variables=["topic"]
)

parser=StrOutputParser()

chain=prompt|model|parser
result=chain.invoke({'topic':'Unemployment'})
print(result)

