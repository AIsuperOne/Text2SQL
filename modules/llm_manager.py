from openai import OpenAI
from config.settings import LLM_CONFIG

class QwenLLM:
    def __init__(self):
        self.client = OpenAI(
            api_key=LLM_CONFIG['api_key'],
            base_url=LLM_CONFIG['base_url']
        )
        self.model = LLM_CONFIG['model']

    def chat(self, messages, **kwargs):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return completion.choices[0].message.content

    def generate_sql(self, system_prompt, user_question):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question},
        ]
        return self.chat(messages)

