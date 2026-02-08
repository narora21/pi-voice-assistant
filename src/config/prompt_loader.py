import logging

logger = logging.getLogger(__name__)

class PromptLoader:
    @staticmethod
    def load_system_prompt():
        with open("./prompts/system_prompt.txt") as f:
            prompt = f.read()
            logger.info(f"System prompt: {prompt}")
            return prompt