"""Run and test every part of the language model management system."""

from conversation import Conversation
from exceptions import InvalidPromptError, ModelNotFoundError
from manager import ModelManager
from models import EmbeddingModel, GPTModel, LanguageModel, LlamaModel, ModelType
from pricing import PricingPlan
from tokenizer import Tokenizer


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 50}\n{title}\n{'=' * 50}")


def run_all_tests() -> None:
    """Create all objects and test every feature of the project."""
    section("1) Object creation")
    standard = PricingPlan(plan_name="Standard", price_per_1000_tokens=0.02)
    premium = PricingPlan(plan_name="Premium", price_per_1000_tokens=0.06)

    gpt = GPTModel(
        model_name="GPT-Model",
        provider="OpenAI",
        max_tokens=4096,
        temperature=0.7,
        api_key="secret-key",
        pricing_plan=standard,
    )
    gpt_large = GPTModel(
        model_name="GPT-Large",
        provider="OpenAI",
        max_tokens=128000,
        temperature=0.7,
        api_key="another-key",
        pricing_plan=premium,
    )
    llama = LlamaModel(
        model_name="Llama-Model",
        provider="Local",
        max_tokens=8192,
        temperature=0.5,
        model_path="/models/llama",
        quantization="4-bit",
        pricing_plan=standard,
    )
    embedder = EmbeddingModel(model_name="Embed-Model", provider="OpenAI", max_tokens=8191)

    gpt.display_info()
    print()
    llama.display_info()

    section("2) Class attribute, class method, static method")
    print("total_models:", LanguageModel.total_models)
    print("get_total_models():", LanguageModel.get_total_models())
    print("validate_temperature(0.7):", LanguageModel.validate_temperature(0.7))
    print("validate_temperature(3.5):", LanguageModel.validate_temperature(3.5))

    section("3) Model execution + tokenizer (composition)")
    print("Tokenizer test:", Tokenizer().count_tokens("Artificial intelligence is powerful"))
    response = gpt.generate_response("Explain machine learning")
    print(response)
    print(f"Input tokens: {gpt.last_input_tokens}")

    section("4) Polymorphism")
    for model in [gpt, llama, embedder]:
        print(f"{type(model).__name__:<15} -> {model.generate_response('Explain OOP')}")

    section("5) Encapsulation")
    gpt.temperature = 1.2
    print("Temperature after setting 1.2:", gpt.temperature)
    try:
        gpt.temperature = 4
    except ValueError as error:
        print(error)
    print("Temperature is still:", gpt.temperature)
    print("Masked API key:", gpt.api_key)

    section("6) Abstraction")
    try:
        LanguageModel("Basic-LLM", "Local", 2048, 0.7)
    except TypeError as error:
        print("Cannot create LanguageModel directly:", error)

    section("7) Magic methods")
    print("str :", gpt)
    print("repr:", repr(gpt))
    print("len :", len(gpt))
    same_gpt = GPTModel("GPT-Model", "OpenAI", 2048, 0.3, "other-key", standard)
    print("gpt == same name & provider:", gpt == same_gpt)
    print("gpt == llama:", gpt == llama)

    section("8) Exception handling (custom exception)")
    for bad_prompt in ["", "    ", 123]:
        try:
            print(gpt.generate_response(bad_prompt))
        except InvalidPromptError as error:
            print(f"InvalidPromptError for {bad_prompt!r}: {error}")

    section("9) Request cost calculation (aggregation)")
    gpt.calculate_request_cost("Explain object-oriented programming in simple words")
    llama.calculate_request_cost("What is Python?")
    print("Both models share the same plan object:", gpt.pricing_plan is llama.pricing_plan)
    del same_gpt
    print("Plan still exists after deleting a model:", standard)

    section("10) Model manager")
    manager = ModelManager()
    for model in [gpt, gpt_large, llama, embedder]:
        manager.add_model(model)
    manager.list_models()
    print("Number of models:", len(manager))
    print("Chat models:", [m.model_name for m in manager.get_models_by_type(ModelType.CHAT)])
    print(
        "Embedding models:",
        [m.model_name for m in manager.get_models_by_type(ModelType.EMBEDDING)],
    )

    section("11) Select the best model")
    print("OpenAI, 3000 tokens  ->", manager.select_model("OpenAI", 3000))
    print("OpenAI, 10000 tokens ->", manager.select_model("OpenAI", 10000))
    gpt_large.is_active = False
    try:
        manager.select_model("OpenAI", 10000)
    except ModelNotFoundError as error:
        print("After deactivating GPT-Large:", error)
    gpt_large.is_active = True

    try:
        manager.get_model("Unknown-Model")
    except ModelNotFoundError as error:
        print(error)

    section("12) Conversation")
    conversation = Conversation(conversation_id=1, model=gpt)
    test_prompts = [
        "Explain object-oriented programming",
        "What is inheritance?",
        "",
        "Give me an example of polymorphism",
    ]
    for prompt in test_prompts:
        try:
            conversation.send_message(prompt)
        except InvalidPromptError as error:
            print(f"Skipped prompt: {error}")
    conversation.display_history()
    print("Messages stored:", len(conversation))




# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def build_system() -> ModelManager:
    """Create the pricing plans and models and return a manager holding them."""
    standard = PricingPlan("Standard", 0.02)
    premium = PricingPlan("Premium", 0.06)
    manager = ModelManager()
    manager.add_model(GPTModel("GPT-Model", "OpenAI", 4096, 0.7, "secret-key", standard))
    manager.add_model(GPTModel("GPT-Large", "OpenAI", 128000, 0.7, "another-key", premium))
    manager.add_model(
        LlamaModel("Llama-Model", "Local", 8192, 0.5, "/models/llama", "4-bit", standard)
    )
    manager.add_model(EmbeddingModel("Embed-Model", "OpenAI", 8191, pricing_plan=standard))
    return manager


"""Interactive operations shared by the terminal menu."""

class InteractiveApp:
    """Keep models, plans and conversation histories across menu actions."""

    def __init__(self) -> None:
        """Build the system once and retain its shared pricing plans."""
        self.manager = build_system()
        self.plans = {}
        for model in self.manager.models.values():
            if model.pricing_plan is not None:
                self.plans[model.pricing_plan.plan_name] = model.pricing_plan
        self.conversations = {}
        self.next_conversation_id = 1

    @staticmethod
    def integer(label: str) -> int:
        """Read a strictly positive whole number."""
        value = input(label).strip()
        if not value.isdigit() or int(value) <= 0:
            raise ValueError("Enter a positive whole number.")
        return int(value)

    @staticmethod
    def text(label: str) -> str:
        """Read a nonempty stripped text value."""
        value = input(label).strip()
        if not value:
            raise ValueError("This field cannot be empty.")
        return value

    def pick_model(self) -> LanguageModel:
        """Retrieve an existing model by the user's numbered choice."""
        names = list(self.manager.models)
        if not names:
            raise ValueError("No models registered. Add a model first.")
        for i, name in enumerate(names, 1):
            model = self.manager.get_model(name)
            print(f"{i}) {model} [{model.model_type.value}, active={model.is_active}]")
        choice = self.integer("Model number: ")
        if choice > len(names):
            raise ValueError("Invalid model number.")
        return self.manager.get_model(names[choice - 1])

    def pick_plan(self) -> PricingPlan | None:
        """Choose a shared pricing plan, or explicitly choose no plan."""
        names = list(self.plans)
        print("0) No pricing plan")
        for i, name in enumerate(names, 1):
            print(f"{i}) {self.plans[name]}")
        value = input("Plan number: ").strip()
        if not value.isdigit() or not 0 <= int(value) <= len(names):
            raise ValueError("Invalid plan number.")
        return None if int(value) == 0 else self.plans[names[int(value) - 1]]

    @staticmethod
    def pick_type() -> ModelType:
        """Read a model type using the existing Enum."""
        return ModelType(input("Type (chat/embedding): ").strip().lower())

    def conversation_for(self, model: LanguageModel) -> Conversation:
        """Return the retained conversation, creating it only on first use."""
        if model.model_name not in self.conversations:
            self.conversations[model.model_name] = Conversation(self.next_conversation_id, model)
            self.next_conversation_id += 1
        return self.conversations[model.model_name]

    def add(self) -> None:
        """Create and register a model of one of the implemented classes."""
        kind = input("Class (gpt/llama/embedding): ").strip().lower()
        if kind not in {"gpt", "llama", "embedding"}:
            raise ValueError("Choose gpt, llama or embedding.")
        name = self.text("Unique model name: ")
        if name in self.manager.models:
            raise ValueError("This model name already exists.")
        provider = self.text("Provider: ")
        capacity = self.integer("Maximum tokens: ")
        temperature = float(input("Temperature (0 to 2): "))
        if not LanguageModel.validate_temperature(temperature):
            raise ValueError("Temperature must be between 0 and 2.")
        plan = self.pick_plan()
        if kind == "gpt":
            model = GPTModel(name, provider, capacity, temperature, "demo-key", plan)
        elif kind == "llama":
            path = self.text("Model path (simulation): ")
            quantization = self.text("Quantization: ")
            model = LlamaModel(name, provider, capacity, temperature, path, quantization, plan)
        else:
            model = EmbeddingModel(name, provider, capacity, temperature, plan)
        self.manager.add_model(model)
        print(f"Added: {model}")

    def remove(self) -> None:
        """Remove the selected registration and its associated session history."""
        model = self.pick_model()
        removed = self.manager.remove_model(model.model_name)
        self.conversations.pop(model.model_name, None)
        print(f"Removed: {removed}. Its conversation history was cleared; pricing plans remain.")

    def chat(self) -> None:
        """Resume a model's history and reject inactive or oversized requests."""
        model = self.pick_model()
        conversation = self.conversation_for(model)
        print(f"Chatting with {model}; type 'back' to return. Stored messages: {len(conversation)}")
        while True:
            prompt = input("You: ")
            if prompt.strip().lower() == "back":
                return
            try:
                if not model.is_active:
                    raise ValueError("This model is inactive. Activate it first.")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise InvalidPromptError("Prompt cannot be empty or whitespace only.")
                tokens = model.tokenizer.count_tokens(prompt)
                if tokens > model.max_tokens:
                    raise ValueError("The prompt exceeds the model's token capacity.")
                response = conversation.send_message(prompt)
                print(f"{model.model_name}: {response}")
                cost = model.pricing_plan.calculate_cost(tokens) if model.pricing_plan is not None else None
                label = f"${cost:.8f}" if cost is not None else "Not configured"
                print(f"Input tokens: {tokens} | Input cost: {label} | Messages stored: {len(conversation)}")
            except (InvalidPromptError, ValueError, TypeError) as error:
                print(f"{type(error).__name__}: {error}")

    def info(self) -> None:
        """Display all information for a model retrieved from the manager."""
        self.pick_model().display_info()

    def temperature(self) -> None:
        """Set temperature through its validating property."""
        model = self.pick_model()
        model.temperature = float(input("Temperature (0 to 2): "))
        print(f"Temperature: {model.temperature}")

    def toggle(self) -> None:
        """Toggle the active property."""
        model = self.pick_model()
        model.is_active = not model.is_active
        print(f"Active: {model.is_active}")

    def cost(self) -> None:
        """Calculate a quote without generating or storing a response."""
        self.pick_model().calculate_request_cost(input("Prompt: "))

    def select(self) -> None:
        """Choose by provider, positive capacity and model type."""
        provider = self.text("Provider: ")
        tokens = self.integer("Required tokens: ")
        kind = self.pick_type()
        model = self.manager.select_model(provider, tokens, kind)
        print(f"Selected: {model} | Maximum tokens: {len(model)}")

    def filtered(self) -> None:
        """List names of models matching a selected type."""
        models = self.manager.get_models_by_type(self.pick_type())
        print([model.model_name for model in models])

    def history(self) -> None:
        """Display a retained conversation without sending a message."""
        self.conversation_for(self.pick_model()).display_history()

    def clear(self) -> None:
        """Clear only the selected model's retained history."""
        self.conversation_for(self.pick_model()).messages.clear()
        print("Conversation cleared.")

    def create_plan(self) -> None:
        """Create a reusable pricing plan with a finite nonnegative price."""
        name = self.text("Unique plan name: ")
        if name in self.plans:
            raise ValueError("This plan name already exists.")
        price = self.price()
        self.plans[name] = PricingPlan(name, price)
        print(f"Created: {self.plans[name]}")

    @staticmethod
    def price() -> float:
        """Read a finite nonnegative unit price."""
        import math
        value = float(input("Price per 1000 tokens: "))
        if not math.isfinite(value) or value < 0:
            raise ValueError("Price must be finite and nonnegative.")
        return value

    def assign_plan(self) -> None:
        """Assign a shared plan or remove a model's current plan."""
        model = self.pick_model()
        model.pricing_plan = self.pick_plan()
        print(f"Plan: {model.pricing_plan if model.pricing_plan is not None else 'Not configured'}")

    def update_plan(self) -> None:
        """Update a shared plan; every referencing model sees the new price."""
        plan = self.pick_plan()
        if plan is None:
            raise ValueError("Select an existing plan to update.")
        plan.price_per_1000_tokens = self.price()
        print(f"Updated shared plan: {plan}")

    def compare(self) -> None:
        """Inspect magic methods and compare two registered models."""
        first = self.pick_model()
        second = self.pick_model()
        print(f"str: {first}\nrepr: {first!r}\nlen: {len(first)}")
        print(f"Equal by name/provider: {first == second}\nSame object: {first is second}")

    def count(self) -> None:
        """Count text tokens without requiring a model or creating history."""
        print("Input tokens:", Tokenizer().count_tokens(input("Text: ")))

    def statistics(self) -> None:
        """Distinguish cumulative creations from current registrations."""
        print("Total models created in this process:", LanguageModel.get_total_models())
        print("Currently registered:", len(self.manager))
        for plan in self.plans.values():
            print(plan)


def main() -> None:
    """Run the complete interactive terminal menu with recoverable errors."""
    app = InteractiveApp()
    actions = {
        "1": ("Chat / resume conversation", app.chat),
        "2": ("Show model information", app.info),
        "3": ("Change temperature", app.temperature),
        "4": ("Calculate request cost", app.cost),
        "5": ("Activate / deactivate model", app.toggle),
        "6": ("Select suitable model (provider / tokens / type)", app.select),
        "7": ("List registered models", app.manager.list_models),
        "8": ("Run original 12 OOP demonstrations", run_all_tests),
        "9": ("Add model", app.add),
        "10": ("Remove model and its history", app.remove),
        "11": ("Filter models by type", app.filtered),
        "12": ("Display conversation history", app.history),
        "13": ("Clear conversation history", app.clear),
        "14": ("Create pricing plan", app.create_plan),
        "15": ("Assign / remove pricing plan", app.assign_plan),
        "16": ("Update shared plan price", app.update_plan),
        "17": ("Compare models / inspect magic methods", app.compare),
        "18": ("Count tokens", app.count),
        "19": ("Model counts and available pricing plans", app.statistics),
    }
    while True:
        print("\n" + "=" * 60)
        print("Language Model Manager")
        print("=" * 60)
        for key, (label, _) in actions.items():
            print(f"{key}) {label}")
        print("0) Exit")
        try:
            choice = input("Choose an option: ").strip()
            if choice == "0":
                print("Goodbye!")
                return
            if choice not in actions:
                print("Invalid option.")
                continue
            actions[choice][1]()
        except (ValueError, TypeError, InvalidPromptError, ModelNotFoundError) as error:
            print(f"{type(error).__name__}: {error}")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return


if __name__ == "__main__":
    main()
