# Project Questions — Answers

**1. What is the difference between a class and an object?**
A class is a blueprint that defines attributes and methods (e.g. `GPTModel`). An object is a concrete instance created from that blueprint with its own data (e.g. `gpt = GPTModel(...)`).

**2. Why did we use `LanguageModel` as an abstract class?**
A "generic" language model has no real way to generate a response; only concrete models do. Making it abstract prevents creating it directly and forces every child class to implement `generate_response()`, while still sharing common code (tokenizer, temperature validation, cost calculation, magic methods).

**3. What is the purpose of `super()`?**
It calls the parent class's method, usually the constructor. Child classes use `super().__init__(...)` so the shared setup (name, provider, tokenizer, counter, …) is written once in the parent instead of being duplicated. It is also used in `display_info()` to extend the parent's output.

**4. What is the difference between inheritance and composition?**
Inheritance is an "is-a" relationship: `GPTModel` **is a** `LanguageModel`. Composition is a "has-a" relationship: a `LanguageModel` **has a** `Tokenizer` that it creates and owns internally.

**5. What is the difference between composition and aggregation?**
In composition the container creates and owns the part, so the part lives and dies with it (`self.tokenizer = Tokenizer()` inside the constructor). In aggregation the part is created outside and passed in, so it can be shared and keeps existing after the container is deleted (the same `PricingPlan` object is shared by `gpt` and `llama`).

**6. Where does polymorphism appear in the project?**
When looping over `[gpt, llama, embedder]` and calling `generate_response()`: the same method name gives a different result for each class. `Conversation` and `ModelManager` also work with any `LanguageModel` without knowing its exact type.

**7. What is the difference between a class attribute and an instance attribute?**
A class attribute (`total_models`) is shared by all objects of the class. An instance attribute (`model_name`, `max_tokens`) belongs to one object only.

**8. Why did we use `@property`?**
To control access to private data. The `temperature` setter validates values (0–2) before saving them, `is_active` accepts only booleans, and `api_key` is read-only and returned masked — while the user still writes simple code like `model.temperature = 1.2`.

**9. When should we use a static method?**
When the function is logically related to the class but needs neither the object (`self`) nor the class (`cls`), e.g. `validate_temperature(value)`, which only checks a number.

**10. When should we use a class method?**
When the method works with class-level data instead of a single object, e.g. `get_total_models()`, which reads the shared `total_models` counter. Class methods are also used as alternative constructors.

**11. Why did we override `generate_response()`?**
Each model produces its response differently (a cloud API call with logging for GPT, a local run for Llama, a vector for embeddings). Overriding lets each class define its own behaviour behind the same method name.

**12. What is the purpose of `__str__` and `__repr__`?**
`__str__` gives a readable description for users (`GPT-Model by OpenAI`), used by `print()`. `__repr__` gives a precise technical description for developers (`GPTModel(model_name='GPT-Model', provider='OpenAI')`), useful for debugging.

**13. Why did we create a custom exception?**
`InvalidPromptError` describes the exact problem clearly, so the caller can catch prompt errors specifically with `except InvalidPromptError` without accidentally catching unrelated errors.

**14. What problem does `ModelManager` solve?**
It keeps all models in one place and centralises adding, removing, finding, listing and selecting the best model (by provider, required tokens and active state), instead of managing separate variables throughout the program.

**15. How could this project be connected to a real API later?**
Only the body of `generate_response()` in each child class would change: instead of returning a simulated string, it would send an HTTP request to the provider's API using the stored API key and return the real answer. Because the rest of the system depends only on the abstract interface, `Conversation`, `ModelManager` and the pricing logic would keep working unchanged.
