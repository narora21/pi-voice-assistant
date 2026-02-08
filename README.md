# pi-voice-assistant
Voice Assistant For Raspberry Pi 5

Stack:
* ollama local models for LLM inference
* fast whisper for speech to text
* open wake word for waking up the program

## Running the project
1. ssh into your raspberry pi
2. clone the repo
3. run the run.sh shell script

## Running unit tests
uv run python -m pytest

### To Do
Transition to a stopped state conditionally
Find better sfx for state transitions
Debug time to first token being slow