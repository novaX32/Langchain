from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from langchain_core.runnables import RunnableBranch,RunnableLambda,RunnablePassthrough

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")

)

model=ChatHuggingFace(llm=llm)
parser=StrOutputParser()

class Topic(BaseModel):
    topic: Literal['maths', 'coding', 'science'] = Field(
        description='Classify the question into maths, coding, or science'
    )

parser2=PydanticOutputParser(pydantic_object=Topic)

prompt1 = PromptTemplate(
    template="""
Classify the following question into one of these categories:
- maths
- coding
- science

Question: {question}

{format_instruction}
""",
    input_variables=["question"],
    partial_variables={"format_instruction": parser2.get_format_instructions()}
)

classify_chain=prompt1|model|parser2

prompt2 = PromptTemplate(
    template="Explain the following maths question step-by-step with proper formulas:\n{question}",
    input_variables=["question"]
)

prompt3 = PromptTemplate(
    template="Debug the following code, explain time complexity, and suggest optimizations:\n{question}",
    input_variables=["question"]
)

prompt4 = PromptTemplate(
    template="Explain the following science question clearly:\n{question}",
    input_variables=["question"]
)


branch_chain = RunnableBranch(
    (
        lambda x: x['topic_obj'].topic == "maths",
        RunnableLambda(lambda x: {"question": x["question"]})
        | prompt2 | model | parser
    ),
    (
        lambda x: x['topic_obj'].topic == "coding",
        RunnableLambda(lambda x: {"question": x["question"]})
        | prompt3 | model | parser
    ),
    (
        lambda x: x['topic_obj'].topic == "science",
        RunnableLambda(lambda x: {"question": x["question"]})
        | prompt4 | model | parser
    ),
    RunnableLambda(lambda x: "Could not classify the question")
)

chain = (
    {
        "topic_obj": classify_chain,        
        "question": RunnablePassthrough()  
    }
    | branch_chain
)

result=chain.invoke({'question':"Whats is python ptogramming language"})
print(result)

