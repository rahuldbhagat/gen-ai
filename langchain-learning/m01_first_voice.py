import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set. See .env.example.")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)


def main():
    question = input("Ask the assistant something: ")
    response = llm.invoke(question)
    print(response.content)


if __name__ == "__main__":
    main()