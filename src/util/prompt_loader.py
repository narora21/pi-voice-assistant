import logging

from datetime import date
from jinja2 import Template

logger = logging.getLogger(__name__)

class PromptLoader:
    @staticmethod
    def load_system_prompt():
        today = date.today()
        date_string = today.strftime("%Y-%m-%d")
        with open("./prompts/system_prompt.txt") as f:
            prompt = f.read()
            prompt_template = Template(prompt)
            full_sys_prompt = prompt_template.render(date=date_string)
            logger.info(f"System prompt: {full_sys_prompt}")
            return full_sys_prompt