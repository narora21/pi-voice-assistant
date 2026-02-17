# pi-voice-assistant
Voice Assistant For Raspberry Pi 5

Stack:
* ollama local models for LLM inference
* fast whisper for speech to text
* open wake word for waking up the program

## Setup

### Ollama

Keep the model loaded in memory indefinitely to avoid cold start latency after idle periods:

```bash
sudo systemctl edit ollama.service
```

Add the following under `[Service]`:

```ini
[Service]
Environment="OLLAMA_KEEP_ALIVE=-1"
```

Then reload and restart:

```bash
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

## Running the project
1. ssh into your raspberry pi
2. clone the repo
3. run the run.sh shell script

## Running unit tests
uv run python -m pytest
