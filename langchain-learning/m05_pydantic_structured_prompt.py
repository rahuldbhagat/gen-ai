from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
class Intent(BaseModel):
    topic: str = Field(description="Topic to explain")
    audience: str = Field(description="Intended audience")
    include_example: bool = Field(description="Whether an example is requested")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
extractor = llm.with_structured_output(Intent)
intent = extractor.invoke("Please explain RAG simply for managers, with an example")
print(intent.model_dump())


prompt = ChatPromptTemplate.from_messages([
    ("system", "Explain the topic to {audience}. Use short sections. Include an example when requested: {include_example}."),
    ("human", "Topic: {topic}")
])
answer = (prompt | llm).invoke(intent.model_dump())
print(answer.content)