import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import numpy as np

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))


def cosine_similarity(vector_a, vector_b):
    a = np.array(vector_a)
    b = np.array(vector_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    text1 = input("Text 1: ")
    text2 = input("Text 2: ")
    vector1 = embeddings.embed_query(text1)
    vector2 = embeddings.embed_query(text2)
    print(f"vector1  has {len(vector1)} numbers.")
    print(f"vector2  has {len(vector2)} numbers.")
    print(f"Cosine similarity: {cosine_similarity(vector1, vector2):.4f}")


if __name__ == "__main__":
    main()