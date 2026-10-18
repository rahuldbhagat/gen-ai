import logging
import sys

from dotenv import load_dotenv
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_postgres import PGEngine
from openai import OpenAIError
from sqlalchemy.exc import SQLAlchemyError

from vector_store import close_engine, create_embeddings, create_engine, open_store, require_env

log = logging.getLogger("app")


def build_chain(engine: PGEngine) -> RunnableWithMessageHistory:
    embeddings = create_embeddings()
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=require_env("OPENAI_API_KEY"))

    # Documents are loaded by ingest.py. Here we only open the stored table and search it.
    retriever = open_store(engine, embeddings).as_retriever(search_kwargs={"k": 3})

    # Step 1: rewrite the question using history. NO {context} here on purpose —
    # this prompt's only job is to produce a standalone question, not an answer.
    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Rewrite the user's latest question as a standalone question, "
                "using the chat history for any missing context. Do not answer it.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    # Step 2: answer using retrieved context. {context} appears here, not above.
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a research assistant. Answer using only the context below. If the
answer isn't in the context, say exactly: "I don't know."

Context:
{context}
""",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    history = ChatMessageHistory()
    return RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv()

    try:
        engine = create_engine()
    except RuntimeError as exc:  # missing environment variable
        log.error("%s", exc)
        return 1

    try:
        try:
            chain_with_history = build_chain(engine)
        except RuntimeError as exc:  # missing env var or missing table
            log.error("%s", exc)
            return 1
        except (SQLAlchemyError, OSError) as exc:
            log.error(
                "Could not connect to Postgres: %s\nIs the container running? Check with: docker compose ps",
                exc,
            )
            return 1

        print("Assistant (with memory + grounding) is ready. Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.strip().lower() in {"exit", "quit"}:
                break
            if not user_input.strip():
                continue
            try:
                response = chain_with_history.invoke(
                    {"input": user_input},
                    {"configurable": {"session_id": "demo-session"}},
                )
            except (SQLAlchemyError, OSError, OpenAIError) as exc:
                # One failed turn should not end the session.
                log.error("That question failed: %s", exc)
                continue
            print("Assistant:", response["answer"])
        return 0
    except (KeyboardInterrupt, EOFError):
        print()
        return 0
    finally:
        close_engine(engine)


if __name__ == "__main__":
    sys.exit(main())