uv init langchain-learning
cd langchain-learning
uv python pin 3.11
uv add langchain langchain-openai langchain-core langchain-community langchain-text-splitters langchain-chroma langchain-classic python-dotenv numpy
uv add langchain-ollama
mkdir langchain_demo
uv add langchain-postgres sqlalchemy openai