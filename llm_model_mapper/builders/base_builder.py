from abc import abstractmethod, ABC


class LLMBaseBuilder(ABC):

    def __init__(self, model_name, api_key, end_point=None, api_version=None):
        self.model_name = model_name
        self.api_key = api_key
        self.end_point = end_point
        self.params = {}
        self.api_version = api_version
        self.llm_class = None
        self.llm_provider = None

    @abstractmethod
    def _construct_llm_params(self):
        pass

    @abstractmethod
    def _validation_parameter(self):
        pass

    @abstractmethod
    def _build_llm_connection(self, additional_params=None):
        pass

    def build(self):
        self._validation_parameter()
        additional_params = self._construct_llm_params()
        return self._build_llm_connection(additional_params)
