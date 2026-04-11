import ollama
import json
import re
from utils.image_prompts import get_image_classification_prompt
from utils.logger import get_logger

logger = get_logger(__name__)


class OllamaProvider:
    def __init__(self, model_name: str, think: bool = False):
        self.model_name = model_name
        self.think = think
        self.client = ollama.Client()
        logger.info("OllamaProvider initialised | model=%s think=%s", self.model_name, self.think)

    def get_response(self, image_data: bytes, message_request: list):
        logger.debug("Sending request to Ollama | model=%s think=%s", self.model_name, self.think)
        response = self.client.chat(model=self.model_name, messages=message_request, think=self.think)
        content = response['message']['content']
        logger.debug("Raw LLM response | model=%s content=%s", self.model_name, content)

        parsed = self._extract_json(content)
        if parsed is not None:
            logger.debug("Parsed JSON from LLM response | result=%s", parsed)
            return parsed

        logger.warning("Returning raw LLM content (no JSON found) | model=%s", self.model_name)
        return content

    def _extract_json(self, content: str):
        # 1. Try direct parse
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            pass

        # 2. Strip markdown code fences (```json ... ``` or ``` ... ```)
        stripped = re.sub(r'```(?:json)?\s*([\s\S]*?)```', r'\1', content).strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        # 3. Grab first {...} block, allowing multiline and nested arrays
        match = re.search(r'\{[\s\S]*\}', content)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError as exc:
                logger.warning("JSON decode failed on LLM response | error=%s raw=%s", exc, match.group())

        return None


# llm_client = OllamaProvider("ministral-3:3b")

# print(json.dumps(llm_client.get_response("data/Sunny-Leone-Nude-hot-39.md.jpg"), indent=2))
