from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion import ChatCompletion

import openai

OLLAMA_BASE_URL = "http://localhost:11434/v1/"

def _get_openai_client(base_url: str | None = None) -> openai.Client:
    api_key = 'ollama'
    return openai.Client(api_key=api_key, base_url=base_url)


def openai_chat_completion(
    messages: list[ChatCompletionMessageParam],
    model: str,
    base_url: str | None = None,
    temperature: float = 0.8,
    **kwargs,
) -> ChatCompletion:
    client = _get_openai_client(base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )
    return response
