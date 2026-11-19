# Language Model Manager

We use Python classes to simulate chat and embedding models, manage shared pricing plans and keep separate conversations for each model. Responses and costs are examples.

## Run locally

Install uv. Open a terminal in the project folder and run:

```sh
uv sync
uv run python app_gradio.py
```

Open the local URL printed in the terminal. For the terminal menu, run `uv run python main.py`. Option 8 runs the twelve OOP demonstrations.

The web interface saves model and pricing changes in model_settings.json. Conversations last for the browser session. ANSWERS.md explains the OOP concepts used in the project.
