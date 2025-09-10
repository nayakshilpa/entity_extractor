from typing import Type, Dict

from .builders import LLMBaseBuilder
from .builders import OpenAILangChainLLMBUilder


class LLMBuilderFactory:
    _registry: Dict[str, Type[LLMBaseBuilder]] = {}

    @classmethod
    def register_builder(cls, framework: str, builder_cls: Type[LLMBaseBuilder]):
        cls._registry[framework.lower()] = builder_cls

    @classmethod
    def get_builder(
        cls,
        llm_provider,
        framework,
        model_name,
        api_key,
        end_point=None,
        api_version=None,
    ) -> LLMBaseBuilder:
        framework = framework.lower()
        llm_provider = llm_provider.lower()
        if f"{framework}_{llm_provider}" not in cls._registry:
            raise ValueError(
                f"No LLM builder available for framework => {framework} & llm_provider => {llm_provider}"
            )

        builder_cls = cls._registry[f"{framework}_{llm_provider}"]
        params = {
            "model_name": model_name,
            "api_key": api_key,
            "end_point": end_point,
            "api_version": api_version,
        }
        return builder_cls(**params)


LLMBuilderFactory.register_builder("openai_langchain", OpenAILangChainLLMBUilder)
