from llm_model_mapper.builders.base_builder import LLMBaseBuilder
from langchain.chat_models import AzureChatOpenAI


class OpenAILangChainLLMBUilder(LLMBaseBuilder):

    def _construct_llm_params(self):
        params = {"temperature": 0, "verbose": False}
        return params

    def _validation_parameter(self):
        if (self.model_name is None) | (self.model_name == ""):
            raise ValueError("Please enter a Validate LLM Model Name")

        if (self.api_key is None) | (self.api_key == ""):
            raise ValueError("Please Enter a Validate api_key for LLM")

    def _build_llm_connection(self, additional_params=None):
        self.llm_class = "AzureChatOpenAI"
        self.llm_provider = "langchain"
        params = {
            "deployment_name": self.model_name,
            "azure_endpoint": self.end_point,
            "api_key": self.api_key,
            "api_version": self.api_version,
        }

        if additional_params:
            self.params = {**additional_params, **params}

        return AzureChatOpenAI(**self.params)
