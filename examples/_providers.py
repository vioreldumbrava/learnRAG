"""Translate the project's AppConfig providers into framework LLM/embeddings.

The whole point of the examples is "same models, different framework": they
read the *same* `config.yaml` the from-scratch app uses (chat/embeddings
provider, model, base_url) and hand those to LangChain / LlamaIndex classes.
This mirrors `src/rag_app/providers/factory.py`, kept in one place so each
example stays about the framework, not about wiring providers.

Imports are lazy (inside each function) so this module imports cleanly whether
or not a given framework extra is installed.
"""

from __future__ import annotations

from rag_app.config import ChatSection, EmbeddingsSection


# A non-empty placeholder: LM Studio's OpenAI-compatible server ignores the
# API key, but the OpenAI SDK insists on one being present.
_DUMMY_KEY = "not-needed"


# ---------------------------------------------------------------------------
# LangChain
# ---------------------------------------------------------------------------

def langchain_chat(cfg: ChatSection):
    if cfg.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=cfg.model, base_url=cfg.base_url, temperature=cfg.temperature,
        )
    # lmstudio -> OpenAI-compatible endpoint (base_url ends in /v1)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=cfg.model,
        base_url=cfg.base_url,
        api_key=_DUMMY_KEY,
        temperature=cfg.temperature,
    )


def langchain_embeddings(cfg: EmbeddingsSection):
    if cfg.provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=cfg.model, base_url=cfg.base_url)
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=cfg.model,
        base_url=cfg.base_url,
        api_key=_DUMMY_KEY,
        # LM Studio embed models aren't OpenAI models, so skip the tiktoken
        # context-length check that assumes they are.
        check_embedding_ctx_length=False,
    )


# ---------------------------------------------------------------------------
# LlamaIndex
# ---------------------------------------------------------------------------

def llamaindex_llm(cfg: ChatSection):
    if cfg.provider == "ollama":
        from llama_index.llms.ollama import Ollama

        return Ollama(
            model=cfg.model,
            base_url=cfg.base_url,
            temperature=cfg.temperature,
            request_timeout=120.0,
        )
    from llama_index.llms.openai_like import OpenAILike

    return OpenAILike(
        model=cfg.model,
        api_base=cfg.base_url,
        api_key=_DUMMY_KEY,
        temperature=cfg.temperature,
        is_chat_model=True,
    )


def llamaindex_embeddings(cfg: EmbeddingsSection):
    if cfg.provider == "ollama":
        from llama_index.embeddings.ollama import OllamaEmbedding

        return OllamaEmbedding(model_name=cfg.model, base_url=cfg.base_url)
    from llama_index.embeddings.openai_like import OpenAILikeEmbedding

    return OpenAILikeEmbedding(
        model_name=cfg.model, api_base=cfg.base_url, api_key=_DUMMY_KEY,
    )
