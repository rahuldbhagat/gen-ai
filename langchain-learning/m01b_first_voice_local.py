from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")


def main():
    question = input("Ask the assistant something (running locally, no API key): ")
    response = llm.invoke(question)
    print(response.content)


if __name__ == "__main__":
    main()