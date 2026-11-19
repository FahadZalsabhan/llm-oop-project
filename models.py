"""Language model classes: the abstract base, its children and the logging mixin."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from exceptions import InvalidPromptError
from pricing import PricingPlan
from tokenizer import Tokenizer


class ModelType(Enum):
    """The kind of output a model produces."""

    CHAT = "chat"
    EMBEDDING = "embedding"


class LoggingMixin:
    """Adds simple logging ability to any class that inherits from it."""

    def log(self, message: str) -> None:
        """Print a log message."""
        print(f"[LOG] {message}")


class LanguageModel(ABC):
    """Abstract base class for every language model in the system."""

    total_models: int = 0

    def __init__(
        self,
        model_name: str,
        provider: str,
        max_tokens: int,
        temperature: float,
        pricing_plan: PricingPlan | None = None,
        model_type: ModelType = ModelType.CHAT,
    ) -> None:
        """Store the model settings and create its internal tokenizer."""
        self.model_name = model_name
        self.provider = provider
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_type = model_type
        self.tokenizer = Tokenizer()
        self.pricing_plan = pricing_plan
        self.last_input_tokens = 0
        self.__is_active = True
        LanguageModel.total_models += 1

    @classmethod
    def get_total_models(cls) -> int:
        """Return how many model objects have been created."""
        return cls.total_models

    @staticmethod
    def validate_temperature(value: float) -> bool:
        """Return True if the temperature is between 0.0 and 2.0."""
        return isinstance(value, (int, float)) and 0.0 <= value <= 2.0

    @property
    def temperature(self) -> float:
        """Return the model temperature."""
        return self.__temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        """Set the temperature after checking it is between 0 and 2."""
        if not LanguageModel.validate_temperature(value):
            raise ValueError("Temperature must be between 0 and 2.")
        self.__temperature = value

    @property
    def is_active(self) -> bool:
        """Return whether the model is available for use."""
        return self.__is_active

    @is_active.setter
    def is_active(self, value: bool) -> None:
        """Activate or deactivate the model."""
        if not isinstance(value, bool):
            raise TypeError("is_active must be True or False.")
        self.__is_active = value

    def _prepare_prompt(self, prompt: str) -> int:
        """Validate the prompt, count its tokens and return the count."""
        if not isinstance(prompt, str):
            raise InvalidPromptError("Prompt must be a string.")
        if prompt == "":
            raise InvalidPromptError("Prompt cannot be empty.")
        if prompt.strip() == "":
            raise InvalidPromptError("Prompt cannot contain only spaces.")
        self.last_input_tokens = self.tokenizer.count_tokens(prompt)
        return self.last_input_tokens

    @abstractmethod
    def generate_response(self, prompt: str) -> str | list[float]:
        """Generate a simulated response for the prompt."""

    def calculate_request_cost(self, prompt: str) -> float:
        """Count the prompt tokens, calculate their cost and display it."""
        if self.pricing_plan is None:
            raise ValueError(f"{self.model_name} has no pricing plan.")
        tokens = self._prepare_prompt(prompt)
        cost = self.pricing_plan.calculate_cost(tokens)
        print(f"Tokens: {tokens} | Plan: {self.pricing_plan.plan_name} | Cost: ${cost:.6f}")
        return cost

    def display_info(self) -> None:
        """Print all model information."""
        print(f"Model Name: {self.model_name}")
        print(f"Provider: {self.provider}")
        print(f"Maximum Tokens: {self.max_tokens}")
        print(f"Temperature: {self.temperature}")
        print(f"Type: {self.model_type.value}")
        print(f"Active: {self.is_active}")
        if self.pricing_plan is not None:
            print(f"Pricing: {self.pricing_plan}")

    def __str__(self) -> str:
        """Return a readable description of the model."""
        return f"{self.model_name} by {self.provider}"

    def __repr__(self) -> str:
        """Return a technical representation of the model."""
        return (
            f"{type(self).__name__}(model_name='{self.model_name}', "
            f"provider='{self.provider}')"
        )

    def __len__(self) -> int:
        """Return the maximum number of tokens."""
        return self.max_tokens

    def __eq__(self, other: object) -> bool:
        """Two models are equal if they have the same name and provider."""
        if not isinstance(other, LanguageModel):
            return NotImplemented
        return (self.model_name, self.provider) == (other.model_name, other.provider)

    def __hash__(self) -> int:
        """Hash the model using the same fields as __eq__."""
        return hash((self.model_name, self.provider))


class GPTModel(LanguageModel, LoggingMixin):
    """A simulated cloud model that is accessed with an API key."""

    def __init__(
        self,
        model_name: str,
        provider: str,
        max_tokens: int,
        temperature: float,
        api_key: str,
        pricing_plan: PricingPlan | None = None,
    ) -> None:
        """Create a GPT model and store its API key privately."""
        super().__init__(model_name, provider, max_tokens, temperature, pricing_plan)
        self.__api_key = api_key

    @property
    def api_key(self) -> str:
        """Return a masked version of the API key."""
        return self.__api_key[:2] + "*" * (len(self.__api_key) - 2)

    def generate_response(self, prompt: str) -> str:
        """Return a simulated GPT API response."""
        self._prepare_prompt(prompt)
        self.log(f"Sending request to {self.model_name}")
        return f"GPT API response to: {prompt}"

    def display_info(self) -> None:
        """Print the model information plus the masked API key."""
        super().display_info()
        print(f"API Key: {self.api_key}")


class LlamaModel(LanguageModel):
    """A simulated model that runs locally from a file on disk."""

    def __init__(
        self,
        model_name: str,
        provider: str,
        max_tokens: int,
        temperature: float,
        model_path: str,
        quantization: str,
        pricing_plan: PricingPlan | None = None,
    ) -> None:
        """Create a Llama model with its path and quantization level."""
        super().__init__(model_name, provider, max_tokens, temperature, pricing_plan)
        self.model_path = model_path
        self.quantization = quantization

    def generate_response(self, prompt: str) -> str:
        """Return a simulated local Llama response."""
        self._prepare_prompt(prompt)
        return f"Local Llama response to: {prompt}"

    def display_info(self) -> None:
        """Print the model information plus path and quantization."""
        super().display_info()
        print(f"Model Path: {self.model_path}")
        print(f"Quantization: {self.quantization}")


class EmbeddingModel(LanguageModel):
    """A simulated model that turns text into an embedding vector."""

    def __init__(
        self,
        model_name: str,
        provider: str,
        max_tokens: int,
        temperature: float = 0.0,
        pricing_plan: PricingPlan | None = None,
    ) -> None:
        """Create an embedding model."""
        super().__init__(
            model_name,
            provider,
            max_tokens,
            temperature,
            pricing_plan,
            model_type=ModelType.EMBEDDING,
        )

    def generate_response(self, prompt: str) -> list[float]:
        """Return a simulated embedding vector for the prompt."""
        self._prepare_prompt(prompt)
        return [0.12, 0.45, 0.88, 0.31]
