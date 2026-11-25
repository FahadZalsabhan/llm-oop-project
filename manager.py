"""Manager that stores and selects language models."""

from __future__ import annotations

from exceptions import ModelNotFoundError
from models import LanguageModel, ModelType


class ModelManager:
    """Keeps a collection of language models organised by name."""

    def __init__(self) -> None:
        """Create a manager with no models."""
        self.models: dict[str, LanguageModel] = {}

    def add_model(self, model: LanguageModel) -> None:
        """Add a model to the manager."""
        if not isinstance(model, LanguageModel):
            raise TypeError("Only LanguageModel objects can be added.")
        if model.model_name in self.models:
            raise ValueError(f"A model named '{model.model_name}' already exists.")
        self.models[model.model_name] = model

    def remove_model(self, model_name: str) -> LanguageModel:
        """Remove a model by name and return it."""
        if model_name not in self.models:
            raise ModelNotFoundError(f"Model '{model_name}' was not found.")
        return self.models.pop(model_name)

    def get_model(self, model_name: str) -> LanguageModel:
        """Return a model by name."""
        if model_name not in self.models:
            raise ModelNotFoundError(f"Model '{model_name}' was not found.")
        return self.models[model_name]

    def list_models(self) -> list[str]:
        """Print and return the names of all models."""
        for name in self.models:
            print(name)
        return list(self.models)

    def get_models_by_type(self, model_type: ModelType) -> list[LanguageModel]:
        """Return all models of the given type (chat or embedding)."""
        return [m for m in self.models.values() if m.model_type is model_type]

    def select_model(
        self,
        provider: str,
        required_tokens: int,
        model_type: ModelType = ModelType.CHAT,
    ) -> LanguageModel:
        """Return the smallest active model from the provider that has enough tokens."""
        if type(required_tokens) is not int or required_tokens <= 0:
            raise ValueError("Required tokens must be a positive integer.")
        candidates = [
            m
            for m in self.models.values()
            if m.provider.lower() == provider.lower()
            and m.max_tokens >= required_tokens
            and m.is_active
            and m.model_type is model_type
        ]
        if not candidates:
            raise ModelNotFoundError(
                f"No active {model_type.value} model from '{provider}' "
                f"supports {required_tokens} tokens."
            )
        return min(candidates, key=lambda m: m.max_tokens)

    def __len__(self) -> int:
        """Return the number of managed models."""
        return len(self.models)

    def __contains__(self, model_name: str) -> bool:
        """Return True if a model with this name is managed."""
        return model_name in self.models
