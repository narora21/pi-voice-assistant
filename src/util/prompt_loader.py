import logging

logger = logging.getLogger(__name__)

class PromptLoader:
    @staticmethod
    def load_system_prompt():
        with open("./prompts/system_prompt.txt") as f:
            prompt = f.read()
            return prompt