import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

document = TextLoader("demo_notes.txt").load()
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = splitter.split_documents(document)
print(f"Loaded 1 document, split into {len(chunks)} chunks.")

db = Chroma.from_documents(chunks, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 2})


def main():
    query = input("Ask about something in demo_notes.txt: ")
    results = retriever.invoke(query)
    for i, doc in enumerate(results, start=1):
        print(f"--- Match {i} ---")
        print(doc.page_content)
        print()


if __name__ == "__main__":
    main()