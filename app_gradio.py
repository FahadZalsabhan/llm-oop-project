"""Local interactive Gradio frontend for the existing OOP project."""

from pathlib import Path
import io
import json
import math
from contextlib import redirect_stdout

import gradio as gr

from main import InteractiveApp, run_all_tests
from models import GPTModel, LlamaModel, EmbeddingModel, LanguageModel, ModelType
from exceptions import InvalidPromptError, ModelNotFoundError
from pricing import PricingPlan
from tokenizer import Tokenizer

NO_PLAN = "(No plan)"


CONFIG = Path(__file__).with_name("model_settings.json")


def snapshot(system):
    models = []
    for model in system.manager.models.values():
        models.append(dict(kind=type(model).__name__, name=model.model_name,
                           provider=model.provider, capacity=model.max_tokens,
                           temperature=model.temperature, active=model.is_active,
                           plan=model.pricing_plan.plan_name if model.pricing_plan else None,
                           path=getattr(model, "model_path", None),
                           quantization=getattr(model, "quantization", None)))
    return {"plans": {name: p.price_per_1000_tokens for name, p in system.plans.items()},
            "models": models}


def restore(system, data):
    system.plans = {n: PricingPlan(n, price) for n, price in data["plans"].items()}
    defaults = dict(system.manager.models)
    system.manager.models.clear()
    for row in data["models"]:
        args = (row["name"], row["provider"], row["capacity"], row["temperature"])
        plan = system.plans[row["plan"]] if row["plan"] is not None else None
        if row["kind"] == "GPTModel":
            model = defaults.get(row["name"])
            if not isinstance(model, GPTModel):
                model = GPTModel(*args, "demo-key", plan)
            model.provider, model.max_tokens, model.temperature = args[1:]
            model.pricing_plan = plan
        elif row["kind"] == "LlamaModel":
            model = LlamaModel(*args, row["path"], row["quantization"], plan)
        else:
            model = EmbeddingModel(*args, plan)
        model.is_active = row["active"]
        system.manager.add_model(model)


def create_session() -> dict:
    """Own an independent manager, plan registry and histories per browser session."""
    system = InteractiveApp()
    saved = CONFIG.read_text(encoding="utf8") if CONFIG.exists() else None
    if saved is not None:
        restore(system, json.loads(saved))
    return {"system": system, "saved": saved, "selected": "GPT-Model", "results": {}}


def ensure(session: dict | None) -> dict:
    """Lazily initialize a browser session."""
    return create_session() if session is None else session


def positive_integer(value: float, field: str) -> int:
    """Reject non-finite, fractional, zero and negative capacities."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 or int(value) != value:
        raise ValueError(f"{field} must be a positive whole number.")
    return int(value)


def text(value: str, field: str) -> str:
    """Validate required text without silently accepting empty names."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} cannot be empty.")
    return value.strip()


def price_value(value: float) -> float:
    """Validate a simulated unit price."""
    if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
        raise ValueError("Price must be finite and nonnegative.")
    return float(value)


def selected_model(session: dict):
    """Resolve the current model through ModelManager."""
    return session["system"].manager.get_model(session["selected"])


def history_messages(conversation) -> list[dict]:
    """Format vectors for display without modifying stored messages."""
    return [{"role": message["role"], "content":
             "```json\n" + json.dumps(message["content"]) + "\n```"
             if isinstance(message["content"], list) else message["content"]}
            for message in conversation.messages]


def refresh(session: dict | None, status: str = "Ready.") -> tuple:
    """Synchronize every shared control after a model or plan mutation."""
    session = ensure(session)
    system = session["system"]
    names = list(system.manager.models)
    if session["selected"] not in names:
        session["selected"] = names[0] if names else None
    model = system.manager.get_model(session["selected"]) if names else None
    rows = [[m.model_name, m.provider, m.model_type.value, m.max_tokens, m.temperature,
             m.is_active, m.pricing_plan.plan_name if m.pricing_plan is not None else "Not configured"]
            for m in system.manager.models.values()]
    plans = [[p.plan_name, p.price_per_1000_tokens,
              ", ".join(m.model_name for m in system.manager.models.values() if m.pricing_plan is p) or "—"]
             for p in system.plans.values()]
    info = "No models registered. Use the Models tab to add one."
    messages = []
    result = "Input tokens: — · Input cost: — · Messages stored: 0"
    if model is not None:
        with redirect_stdout(io.StringIO()) as out:
            model.display_info()
        info = out.getvalue()
        conversation = system.conversations.get(model.model_name)
        messages = history_messages(conversation) if conversation else []
        last = session["results"].get(model.model_name)
        token_label = str(last[0]) if last else "—"
        cost_label = (f"${last[1]:.8f}" if last[1] is not None else "Not configured") if last else "—"
        result = f"Input tokens (last message): {token_label} · Input cost: {cost_label} · Messages stored: {len(messages)}"
    return (session,
            gr.Dropdown(choices=names, value=model.model_name if model else None),
            messages, info,
            gr.Slider(value=float(model.temperature) if model else 0.7, interactive=model is not None),
            gr.Checkbox(value=model.is_active if model else False, interactive=model is not None),
            rows,
            gr.Dropdown(choices=[NO_PLAN]+list(system.plans), value=model.pricing_plan.plan_name if model and model.pricing_plan is not None else NO_PLAN),
            plans,
            gr.Dropdown(choices=names, value=names[0] if names else None),
            status, result)


def switch_model(name: str, session: dict | None) -> tuple:
    """Switch the view without erasing any model's history."""
    session = ensure(session)
    session["selected"] = name
    return refresh(session, "Model selected. Its conversation history is preserved.")


def dispatch(action: str, session: dict | None, *args) -> tuple:
    """Execute explicit actions while keeping validation errors in the interface."""
    session = ensure(session)
    system = session["system"]
    try:
        editing = action in {"add", "remove", "settings", "create_plan", "update_plan"}
        if editing:
            disk = CONFIG.read_text(encoding="utf8") if CONFIG.exists() else None
            if disk != session["saved"]:
                raise ValueError("Settings changed in another tab. Refresh before editing.")
            import copy
            session = copy.deepcopy(session)
            system = session["system"]
        if action == "send":
            prompt = args[0]
            model = selected_model(session)
            if not model.is_active:
                raise ValueError("Model is inactive. Activate it in Settings.")
            if not isinstance(prompt, str) or not prompt.strip():
                raise InvalidPromptError("Prompt cannot be empty or whitespace only.")
            if model.tokenizer.count_tokens(prompt) > model.max_tokens:
                raise ValueError("The prompt exceeds this model's token capacity.")
            with redirect_stdout(io.StringIO()):
                conversation = system.conversation_for(model)
                conversation.send_message(prompt)
                tokens = model.last_input_tokens
                cost = model.pricing_plan.calculate_cost(tokens) if model.pricing_plan is not None else None
            session["results"][model.model_name] = (tokens, cost)
            status = "Message sent. Input cost is an example, not a real charge."
        elif action == "clear":
            model = selected_model(session)
            if model.model_name in system.conversations:
                system.conversations[model.model_name].messages.clear()
            session["results"].pop(model.model_name, None)
            status = "Selected conversation cleared. Other conversations are unchanged."
        elif action == "settings":
            temperature, active, plan_name = args
            model = selected_model(session)
            if not LanguageModel.validate_temperature(temperature):
                raise ValueError("Temperature must be between 0 and 2.")
            plan = None if plan_name == NO_PLAN else system.plans[plan_name]
            model.temperature = temperature
            model.is_active = bool(active)
            model.pricing_plan = plan
            status = "Model settings saved. Temperature does not change simulated responses."
        elif action == "add":
            kind, name, provider, capacity, temperature, plan_name, path, quantization = args
            name, provider = text(name, "Model name"), text(provider, "Provider")
            if name in system.manager.models:
                raise ValueError("A model with this name already exists.")
            capacity = positive_integer(capacity, "Maximum tokens")
            if not LanguageModel.validate_temperature(temperature):
                raise ValueError("Temperature must be between 0 and 2.")
            plan = None if plan_name == NO_PLAN else system.plans[plan_name]
            if kind == "GPT":
                model = GPTModel(name, provider, capacity, temperature, "demo-key", plan)
            elif kind == "Llama":
                model = LlamaModel(name, provider, capacity, temperature, text(path, "Model path"), text(quantization, "Quantization"), plan)
            elif kind == "Embedding":
                model = EmbeddingModel(name, provider, capacity, temperature, plan)
            else:
                raise ValueError("Select a valid model class.")
            system.manager.add_model(model)
            session["selected"] = name
            status = f"Added {name}. A new model object and its own tokenizer were created."
        elif action == "remove":
            if not args[0]:
                raise ValueError("Check the confirmation box before removing a model.")
            model = selected_model(session)
            system.manager.remove_model(model.model_name)
            system.conversations.pop(model.model_name, None)
            session["results"].pop(model.model_name, None)
            status = f"Removed {model.model_name} and its history. Pricing plans are retained."
        elif action == "create_plan":
            name, price = text(args[0], "Plan name"), price_value(args[1])
            if name == NO_PLAN or name in system.plans:
                raise ValueError("Choose a unique plan name.")
            system.plans[name] = PricingPlan(name, price)
            status = f"Created plan {name}. Select it in Settings or when adding a model."
        elif action == "update_plan":
            name, price = args[0], price_value(args[1])
            if name == NO_PLAN:
                raise ValueError("Choose a pricing plan first.")
            system.plans[name].price_per_1000_tokens = price
            status = f"Updated shared plan {name}. All linked models use the new price for future requests."
        else:
            raise ValueError("Unknown action.")
        if editing:
            saved = json.dumps(snapshot(system), ensure_ascii=False, indent=2)
            temporary = CONFIG.with_suffix(".tmp")
            temporary.write_text(saved, encoding="utf8")
            temporary.replace(CONFIG)
            session["saved"] = saved
        return refresh(session, status)
    except (ValueError, TypeError, KeyError, InvalidPromptError, ModelNotFoundError) as error:
        return refresh(session, f"{type(error).__name__}: {error}")


def evaluate(kind: str, session: dict | None, *args) -> str:
    """Run read-oriented tools and return formatted results."""
    session = ensure(session)
    system = session["system"]
    try:
        if kind == "select":
            provider, tokens, model_type = args
            model = system.manager.select_model(text(provider, "Provider"), positive_integer(tokens, "Required tokens"), ModelType(model_type))
            return f"Selected: {model}\nType: {model.model_type.value}\nMaximum tokens: {len(model)}\nThis result does not change the model selected above."
        if kind == "filter":
            return "\n".join(m.model_name for m in system.manager.get_models_by_type(ModelType(args[0]))) or "No matching models."
        if kind == "cost":
            model = selected_model(session)
            with redirect_stdout(io.StringIO()):
                cost = model.calculate_request_cost(args[0])
            return f"Input tokens: {model.last_input_tokens}\nInput cost: ${cost:.8f}\nNo response generated and no message stored."
        if kind == "tokens":
            return f"Input tokens: {Tokenizer().count_tokens(args[0])} (one whitespace-separated word per token)"
        if kind == "compare":
            first, second = selected_model(session), system.manager.get_model(args[0])
            return f"str: {first}\nrepr: {first!r}\nlen: {len(first)}\nEqual by name/provider (==): {first == second}\nSame object (is): {first is second}\nSame tokenizer: {first.tokenizer is second.tokenizer}\nSame pricing-plan object: {first.pricing_plan is second.pricing_plan if first.pricing_plan is not None and second.pricing_plan is not None else 'Not applicable: missing plan'}\nRegistered models: {len(system.manager)}\nModels created in this Python process: {LanguageModel.get_total_models()}"
        if kind == "tests":
            with redirect_stdout(io.StringIO()) as out:
                run_all_tests()
            return "Original demonstrations (expected errors are handled). total_models is cumulative across sessions and runs.\n\n" + out.getvalue()
        raise ValueError("Unknown tool.")
    except (ValueError, TypeError, KeyError, InvalidPromptError, ModelNotFoundError) as error:
        return f"{type(error).__name__}: {error}"


CSS = """
.gradio-container {max-width: 1240px !important; margin: auto;}
#hero {padding: 24px 28px; border-radius: 18px; background: linear-gradient(115deg,#172554,#312e81); color: white; margin-bottom: 16px;}
#hero h1, #hero p {color: white;}
#metrics {padding: 12px 18px; border: 1px solid #94a3b8; border-radius: 12px;}
"""


def build_app():
    """Arrange a full interactive interface and bind each operation to OOP objects."""
    with gr.Blocks(title="Language Model Manager") as app:
        session = gr.State(None)
        gr.Markdown("# Language Model Manager\nInteractive OOP workspace · Four model types of objects, shared plans, independent conversations.", elem_id="hero")
        gr.Markdown("**Simulation:** no real API calls. Replies, vectors and prices are examples. Temperature is stored only. Models, settings and pricing plans are saved locally. Conversations last for this session.")
        with gr.Row():
            model_choice = gr.Dropdown([], label="Current model", interactive=True, scale=2)
            status = gr.Textbox(label="Action result", interactive=False, scale=4)
        with gr.Tabs():
            with gr.Tab("💬 Conversation"):
                metrics = gr.Markdown("Input tokens: — · Input cost: — · Messages stored: 0", elem_id="metrics")
                chat = gr.Chatbot(label="Conversation", height=400)
                prompt = gr.Textbox(label="Your message", placeholder="Explain OOP", lines=2)
                with gr.Row():
                    send = gr.Button("Send message", variant="primary")
                    clear = gr.Button("Clear selected history")
                gr.Markdown("Each model keeps its own history. Previous messages are displayed but are not sent as context to the simulated model.")
            with gr.Tab("⚙️ Settings"):
                with gr.Row():
                    info = gr.Textbox(label="Model information", lines=11, interactive=False)
                    with gr.Column():
                        temperature = gr.Slider(0, 2, value=0.7, step=0.1, label="Temperature")
                        active = gr.Checkbox(value=True, label="Model active")
                        plan_choice = gr.Dropdown([NO_PLAN], value=NO_PLAN, label="Shared pricing plan", interactive=True)
                        save = gr.Button("Save settings", variant="primary")
                        gr.Markdown("Click Save to apply temperature, activation and plan changes. New requests use the updated settings.")
            with gr.Tab("🧩 Models"):
                model_table = gr.Dataframe(headers=["Name", "Provider", "Type", "Capacity", "Temperature", "Active", "Plan"], interactive=False, label="Registered models")
                with gr.Accordion("Add a model", open=True):
                    with gr.Row():
                        add_kind = gr.Dropdown(["GPT", "Llama", "Embedding"], value="GPT", label="Class")
                        add_name = gr.Textbox(label="Unique model name")
                        add_provider = gr.Textbox(label="Provider", value="OpenAI")
                    with gr.Row():
                        add_capacity = gr.Number(value=4096, label="Maximum tokens")
                        add_temp = gr.Number(value=0.7, label="Initial temperature (0–2)")
                    add_path = gr.Textbox(label="Model path (Llama only)", value="/models/llama")
                    add_quant = gr.Textbox(label="Quantization (Llama only)", value="4-bit")
                    gr.Markdown("The new model uses the pricing plan currently selected in Settings, or no plan. GPT keys are simulated automatically.")
                    add_button = gr.Button("Add model", variant="primary")
                with gr.Accordion("Remove current model", open=False):
                    confirm = gr.Checkbox(label="I confirm removal of the current model and its conversation history")
                    remove_button = gr.Button("Remove model", variant="stop")
            with gr.Tab("💳 Pricing"):
                plan_table = gr.Dataframe(headers=["Plan", "Price / 1,000 input tokens", "Linked models"], interactive=False)
                with gr.Row():
                    with gr.Column():
                        new_plan = gr.Textbox(label="New plan name")
                        new_price = gr.Number(label="Price per 1,000 tokens", value=0.02)
                        create_plan = gr.Button("Create plan", variant="primary")
                    with gr.Column():
                        gr.Markdown("To update a plan, select it in Settings. This changes the shared object for every linked model.")
                        updated_price = gr.Number(label="New shared price", value=0.02)
                        update_plan = gr.Button("Update selected plan price")
            with gr.Tab("🔎 Selection & tools"):
                with gr.Row():
                    with gr.Column():
                        provider = gr.Textbox(label="Required provider", value="OpenAI")
                        capacity = gr.Number(label="Required tokens", value=3000)
                        kind = gr.Dropdown(["chat", "embedding"], value="chat", label="Required type")
                        select = gr.Button("Find suitable model", variant="primary")
                        filter_button = gr.Button("List models of selected type")
                        selection_result = gr.Textbox(label="Selection result", lines=5, interactive=False)
                    with gr.Column():
                        quote_prompt = gr.Textbox(label="Text for token count / input cost", lines=3)
                        count = gr.Button("Count tokens")
                        quote = gr.Button("Calculate cost with current model")
                        quote_result = gr.Textbox(label="Tool result", lines=4, interactive=False)
            with gr.Tab("🧪 OOP demonstrations"):
                second = gr.Dropdown([], label="Compare current model with", interactive=True)
                compare = gr.Button("Inspect & compare")
                comparison = gr.Textbox(label="Magic methods and relationships", lines=10, interactive=False)
                gr.Markdown("The creation counter is shared across this Python process, including browser sessions and repeated demonstrations. Registered count belongs to this session's manager.")
                tests = gr.Button("Run original 12 demonstrations")
                test_output = gr.Textbox(label="Demonstration output", lines=18, interactive=False)

        outputs = [session, model_choice, chat, info, temperature, active, model_table, plan_choice, plan_table, second, status, metrics]
        shared = {"concurrency_id": "project-events", "concurrency_limit": 1}
        app.load(refresh, [session], outputs, **shared)
        model_choice.input(switch_model, [model_choice, session], outputs, **shared)
        def bind(button, action, inputs):
            button.click(lambda s, *values: dispatch(action, s, *values), [session]+inputs, outputs, **shared)
        bind(send, "send", [prompt])
        bind(clear, "clear", [])
        bind(save, "settings", [temperature, active, plan_choice])
        bind(add_button, "add", [add_kind, add_name, add_provider, add_capacity, add_temp, plan_choice, add_path, add_quant])
        bind(remove_button, "remove", [confirm])
        bind(create_plan, "create_plan", [new_plan, new_price])
        bind(update_plan, "update_plan", [plan_choice, updated_price])
        select.click(lambda s, p, n, k: evaluate("select", s, p, n, k), [session, provider, capacity, kind], selection_result, **shared)
        filter_button.click(lambda s, k: evaluate("filter", s, k), [session, kind], selection_result, **shared)
        count.click(lambda s, p: evaluate("tokens", s, p), [session, quote_prompt], quote_result, **shared)
        quote.click(lambda s, p: evaluate("cost", s, p), [session, quote_prompt], quote_result, **shared)
        compare.click(lambda s, n: evaluate("compare", s, n), [session, second], comparison, **shared)
        tests.click(lambda s: evaluate("tests", s), [session], test_output, **shared)
    return app


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", share=False, css=CSS, theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"))
