import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

questions_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""You are a research assistant.
List exactly 3 key questions a newcomer should ask to understand: {topic}
Respond with a numbered list only, no other text.
""",
)

answers_prompt = PromptTemplate(
    input_variables=["questions"],
    template="""You are a research assistant.
Answer each of these questions in 2-3 sentences:
{questions}
""",
)

research_chain = (
    questions_prompt
    | llm
    | StrOutputParser()
    | (lambda questions_text: {"questions": questions_text})
    | answers_prompt
    | llm
)


def main():
    topic = input("Topic: ")
    response = research_chain.invoke({"topic": topic})
    print(response.content)


if __name__ == "__main__":
    main()