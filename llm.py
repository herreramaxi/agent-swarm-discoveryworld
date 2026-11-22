from typing import Literal

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv(override=True)

models: dict[str, BaseChatModel] = {
    # Groq - Open-weight models
    "openai/gpt-oss-20b": ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
    ),
    "openai/gpt-oss-120b": ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
    ),
    "qwen/qwen3.8-27b": ChatGroq(
        model="qwen/qwen3.8-27b",
        temperature=0,
    ),  
    # Agent: OpenAI model
    "gpt-5-nano": ChatOpenAI(
        model="gpt-5-nano",
        reasoning_effort="low",
        max_retries=3,
        timeout=60,
    ),
    #expensive
    "gpt-5.5": ChatOpenAI(
        model="gpt-5.5",
        reasoning_effort="low",
        max_retries=3,
        timeout=60,
    ),
    "gpt-6-luna": ChatOpenAI(
        model="gpt-6-luna",
        reasoning_effort="medium",
        max_retries=3,
        timeout=60,
    ),
    "gpt-6.2-sol": ChatOpenAI(
        model="gpt-6.2-sol",
        reasoning_effort="medium",
        max_retries=3,
        timeout=60,
    ),
    "gpt-6.2-astra": ChatOpenAI(
        model="gpt-6.2-astra",
        reasoning_effort="medium",
        max_retries=3,
        timeout=60,
    ),
    "gpt-6-astra": ChatOpenAI(
        model="gpt-6-astra",
        reasoning_effort="low",
        max_retries=3,
        timeout=60,
    ),
    # Agent: fast DeepSeek model
     "deepseek-flash": ChatDeepSeek(
            model="deepseek-flash",
            temperature=0,
            max_retries=3,
            timeout=60,
        ),
    "deepseek-flash-low": ChatDeepSeek(
        model="deepseek-flash",
        reasoning_effort="low",
        temperature=0,
        max_retries=3,
        timeout=60,
    ),   
    # Fixed evaluation judge
    "deepseek-v4-pro": ChatDeepSeek(
        model="deepseek-v4-pro",
        temperature=0,
        max_retries=3,
        timeout=60,
        #https://github.com/deepseek-ai/DeepSeek-V3/issues/1376
        extra_body={"thinking": {"type": "disabled"}},
    ),
}

ReasoningSummary = Literal["auto", "concise", "detailed"]

def get_llm(
    model_name: str,
    *,
    reasoning_effort: str | None = None,
    reasoning_summary: ReasoningSummary | None = None,
) -> BaseChatModel:
    """Get a registered model with optional OpenAI reasoning settings."""
    model = models[model_name]
    if reasoning_effort is None and reasoning_summary is None:
        return model
    if not isinstance(model, ChatOpenAI):
        raise ValueError("Reasoning overrides in get_llm are only supported for OpenAI models.")
    if model.reasoning is None and reasoning_summary is None:
        return model.model_copy(update={"reasoning_effort": reasoning_effort})

    reasoning = dict(model.reasoning or {})
    reasoning["effort"] = reasoning_effort or reasoning.get("effort") or model.reasoning_effort
    if reasoning_summary is not None:
        reasoning["summary"] = reasoning_summary
    return model.model_copy(update={
        "reasoning_effort": None,
        "reasoning": reasoning,
        "output_version": "responses/v1",
    })


# Keep the model names used by existing notebooks.
models.update({
    "gpt-6-luna-none": get_llm("gpt-6-luna", reasoning_effort="none"),
    "gpt-6-luna-low": get_llm("gpt-6-luna", reasoning_effort="low"),
    "gpt-6-luna-medium": get_llm("gpt-6-luna", reasoning_summary="auto"),
    "gpt-6.2-sol-medium": models["gpt-6.2-sol"],
    "gpt-6.2-astra-medium": models["gpt-6.2-astra"],
    "gpt-6.2-astra-max": get_llm("gpt-6.2-astra", reasoning_effort="max"),
})
