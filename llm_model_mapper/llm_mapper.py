from .llm_builder_factory import LLMBuilderFactory


class LLMMapper:
    def __init__(
        self,
        llm_provider,
        framework,
        model_name,
        api_key,
        end_point=None,
        api_version=None,
    ):
        self.llm_provider = llm_provider
        self.framework = framework
        self.model_name = model_name
        self.api_key = api_key
        self.end_point = end_point
        self.api_version = api_version
        self.llm_class = None
        self.model = None

    def construct_llm(self):
        builder = LLMBuilderFactory.get_builder(
            llm_provider=self.llm_provider,
            framework=self.framework,
            model_name=self.model_name,
            api_key=self.api_key,
            end_point=self.end_point,
            api_version=self.api_version,
        )

        self.model = builder.build()
        self.llm_class = builder.llm_class
        self.llm_provider = builder.llm_provider
        return self
