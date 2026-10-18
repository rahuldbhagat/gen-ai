import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

prompt_template = PromptTemplate(
    input_variables=["topic", "audience"],
    template="""You are a careful research assistant.
Explain the following topic: {topic}
Tailor the explanation for this audience: {audience}

If the topic is not a real, verifiable subject, respond with exactly this sentence
and nothing else: "I don't have reliable information
Otherwise, keep the explanation to 1 short paragraphs.
""",
)


def main():
    topic = input("Topic: ")
    audience = input("Audience (e.g. '10-year-old', 'expert'): ")
    prompt = prompt_template.format(topic=topic, audience=audience)
    response = llm.invoke(prompt)
    print(response.content)


if __name__ == "__main__":
    main()