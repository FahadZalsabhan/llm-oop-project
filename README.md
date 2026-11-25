# Language Model Manager

A local Python application for exploring model selection, conversation history and shared pricing plans. The same model classes power a Gradio interface and a terminal menu. Chat replies and embedding vectors are simulated, so you can run the project without model downloads or API credentials.

## Run locally

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run these commands from the repository folder. The project targets Python 3.14 and includes a dependency lockfile.

```sh
uv sync --locked
uv run python app_gradio.py
```

Open the local URL printed by Gradio. To use the terminal menu, run `uv run python main.py`. Choose option 8 to walk through the twelve OOP demonstrations.

## Try model selection

In **Selection & tools**, select provider **OpenAI** and type **chat**. With the default models active, a request for 3,000 tokens selects `GPT-Model`. A request for 10,000 selects `GPT-Large`. The manager filters by provider, model type, active status and capacity, then picks the smallest model that fits.

In **Conversation**, send a message and inspect the word count and estimated input cost. Models can share a pricing plan. Each model has its own conversation history within the browser session. Model and pricing changes made in the web interface are saved locally in `model_settings.json`.

## Project layout

| File | Responsibility |
| --- | --- |
| `models.py` | Abstract model class and simulated GPT, Llama and embedding implementations |
| `manager.py` | Model registration, lookup and selection |
| `conversation.py` | Message history for one model |
| `pricing.py`, `tokenizer.py` | Shared pricing plans and whitespace-based token counting |
| `main.py`, `app_gradio.py` | Terminal menu and web interface |

## Tests

```sh
uv run python -m unittest discover -s tests -v
```

The tests cover capacity boundaries, disabled models, invalid requests, embedding selection and API key display. The course notes explain how the implementation uses inheritance, composition and aggregation. [OOP notes](ANSWERS.md)
