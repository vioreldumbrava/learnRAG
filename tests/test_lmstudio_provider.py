"""Tests for LM Studio provider auto-loading and retry mechanisms."""

import pytest
from unittest.mock import MagicMock, patch
from openai import OpenAIError, APIConnectionError
from rag_app.providers.lmstudio_provider import LmStudioEmbeddingProvider, LmStudioChatProvider
from rag_app.providers.base import ProviderError
from rag_app.models import ChatMessage


def test_lmstudio_embedding_provider_autoload_success():
    """Test that if embeddings.create fails with OpenAIError, the model is auto-loaded and retried successfully."""
    # Set up client mock
    mock_client = MagicMock()
    
    # First call to create fails with OpenAIError, second succeeds
    mock_embeddings = MagicMock()
    mock_embeddings.create.side_effect = [
        OpenAIError("Model not loaded"),
        MagicMock(data=[MagicMock(embedding=[0.1, 0.2, 0.3])])
    ]
    mock_client.embeddings = mock_embeddings

    provider = LmStudioEmbeddingProvider(model="test-embed-model", base_url="http://localhost:1234/v1")
    provider._client = mock_client

    # Mock httpx.Client to simulate successful model load (returns 200)
    mock_response = MagicMock(status_code=200)
    
    with patch("httpx.Client") as mock_client_class:
        mock_http_client = MagicMock()
        mock_http_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_http_client
        
        embeddings = provider.embed_texts(["hello"])
        assert embeddings == [[0.1, 0.2, 0.3]]
        
        # Verify load was called with the correct model and URL
        mock_http_client.post.assert_any_call("http://localhost:1234/api/v1/models/load", json={"model": "test-embed-model"})


def test_lmstudio_chat_provider_autoload_success():
    """Test that if chat.completions.create fails with OpenAIError, the model is auto-loaded and retried successfully."""
    mock_client = MagicMock()
    
    # First call fails with OpenAIError, second succeeds
    mock_chat = MagicMock()
    mock_chat.create.side_effect = [
        OpenAIError("Model not loaded"),
        MagicMock(choices=[MagicMock(message=MagicMock(content="Hi there!"))])
    ]
    mock_client.chat.completions = mock_chat

    provider = LmStudioChatProvider(model="test-chat-model", base_url="http://localhost:1234/v1")
    provider._client = mock_client

    mock_response = MagicMock(status_code=200)
    
    with patch("httpx.Client") as mock_client_class:
        mock_http_client = MagicMock()
        mock_http_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_http_client
        
        reply = provider.generate([ChatMessage(role="user", content="hello")])
        assert reply == "Hi there!"
        
        # Verify load was called with the correct model and URL
        mock_http_client.post.assert_any_call("http://localhost:1234/api/v1/models/load", json={"model": "test-chat-model"})


def test_lmstudio_autoload_fails_propagates_original_error():
    """Test that if auto-load retry also fails, the exception is raised as ProviderError."""
    mock_client = MagicMock()
    
    mock_embeddings = MagicMock()
    # Both calls fail
    mock_embeddings.create.side_effect = [
        OpenAIError("Model not loaded"),
        OpenAIError("Still not loaded")
    ]
    mock_client.embeddings = mock_embeddings

    provider = LmStudioEmbeddingProvider(model="test-embed-model", base_url="http://localhost:1234/v1")
    provider._client = mock_client

    # Simulating API load endpoint returning 500 error
    mock_response = MagicMock(status_code=500, text="Internal error")
    
    with patch("httpx.Client") as mock_client_class:
        mock_http_client = MagicMock()
        mock_http_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_http_client
        
        with pytest.raises(ProviderError) as exc_info:
            provider.embed_texts(["hello"])
            
        assert "LM Studio embeddings request failed" in str(exc_info.value)


def test_lmstudio_embedding_provider_autoload_fallback_self_heal():
    """Test that if the requested model is not downloaded, the provider falls back to the first available model of the same type."""
    mock_client = MagicMock()
    
    # First call fails with OpenAIError, second succeeds with the fallback model
    mock_embeddings = MagicMock()
    mock_embeddings.create.side_effect = [
        OpenAIError("Model not loaded"),
        MagicMock(data=[MagicMock(embedding=[0.5, 0.6, 0.7])])
    ]
    mock_client.embeddings = mock_embeddings

    # Configured with a non-existent model
    provider = LmStudioEmbeddingProvider(model="non-existent-model", base_url="http://localhost:1234/v1")
    provider._client = mock_client

    # Mock native models list to return a list containing an embedding model "nomic-embed"
    mock_get_response = MagicMock(status_code=200)
    mock_get_response.json.return_value = {
        "models": [
            {"key": "some-llm", "type": "llm"},
            {"key": "nomic-embed", "type": "embedding"}
        ]
    }
    
    mock_post_response = MagicMock(status_code=200)
    
    with patch("httpx.Client") as mock_client_class:
        mock_http_client = MagicMock()
        mock_http_client.get.return_value = mock_get_response
        mock_http_client.post.return_value = mock_post_response
        mock_client_class.return_value.__enter__.return_value = mock_http_client
        
        embeddings = provider.embed_texts(["hello"])
        assert embeddings == [[0.5, 0.6, 0.7]]
        
        # Verify provider self-healed its model name
        assert provider.model_name == "nomic-embed"
        
        # Verify load was called with the fallback model
        mock_http_client.post.assert_any_call("http://localhost:1234/api/v1/models/load", json={"model": "nomic-embed"})

