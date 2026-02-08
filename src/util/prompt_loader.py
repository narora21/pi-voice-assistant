import logging

from datetime import datetime
from jinja2 import Template

logger = logging.getLogger(__name__)

class PromptLoader:
    @staticmethod
    def load_system_prompt() -> str:
        with open("./prompts/system_prompt.txt") as f:
            prompt = f.read()
            return prompt
    
    @staticmethod
    def _load_system_reminder(reminder_name, **kwargs) -> str | None:
        try:
            with open(f"./prompts/system_reminders/{reminder_name}.txt") as f:
                template = Template(f.read())
                reminder = template.render(**kwargs)
                return reminder
        except Exception as e:
            logger.error(f"Failed to load system reminder: {reminder_name}")
            return None
    
    @staticmethod 
    def get_system_reminders() -> list[str]:
        try:
            current_date = datetime.now().strftime("%B %d, %Y")
            reminders = [
                PromptLoader._load_system_reminder("date_reminder", date=current_date)
            ]
            sanitized_reminders = filter(lambda r: r is not None, reminders)
            return list(map(lambda r: f"<system-reminder>{r}</system-reminder>", sanitized_reminders))
        except Exception as e:
            logger.error(f"Error while loading system reminders: {e}")
            return []
