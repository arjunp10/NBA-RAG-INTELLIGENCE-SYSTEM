"""Builds a LangChain RetrievalQA chain using Gemini 1.5 Flash and a given retriever."""

import logging
import os

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.schema.retriever import BaseRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

from nba_types import GEMINI_MODEL

load_dotenv()

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """You are an NBA analyst. Use only the context below to answer the question.
If the context does not contain enough information, say so clearly — do not make up facts.

Context:
{context}

Question: {question}

Answer:"""


def build_chain(retriever: BaseRetriever) -> RetrievalQA:
    """Construct a RetrievalQA chain backed by Gemini 1.5 Flash.

    Args:
        retriever: A LangChain retriever that will supply context chunks.

    Returns:
        A RetrievalQA chain that returns the answer and source documents.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your .env file.")

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=0.0,
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": _build_prompt(),
        },
    )

    logger.info("Built RetrievalQA chain with model '%s'", GEMINI_MODEL)
    return chain


def _build_prompt():
    """Construct the PromptTemplate used inside the chain."""
    from langchain.prompts import PromptTemplate

    return PromptTemplate(
        input_variables=["context", "question"],
        template=_PROMPT_TEMPLATE,
    )
