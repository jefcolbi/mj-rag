from itertools import cycle
from typing import List, Union
from litellm import completion, RateLimitError
from litellm.types.utils import ModelResponse

from mj_rag.interfaces import LLMServiceInterface


class RotatingList:
    def __init__(self, initial: list):
        self._initial: list = initial
        self._current = None
        self._cycle = cycle(self._initial)

    @property
    def current(self):
        if self._current:
            return self._current

        self._current = next(self._cycle)
        return self._current

    def next(self):
        self._current = next(self._cycle)

    @property
    def max_iter(self) -> int:
        return len(self._initial)


class LiteLLMService(LLMServiceInterface):
    def __init__(
        self,
        provider: str,
        model: str,
        api_keys_rotator: Union[list, RotatingList],
        base_url: str = None,
    ):
        self.provider = provider
        self.model = model
        self.api_keys_rotator: RotatingList = (
            api_keys_rotator
            if isinstance(api_keys_rotator, RotatingList)
            else RotatingList(api_keys_rotator)
        )
        self.base_url: str = base_url

    def complete_messages(self, messages: List[dict], **kwargs) -> str:
        last_error = None
        try:
            for i in range(self.api_keys_rotator.max_iter):
                try:
                    resp: ModelResponse = completion(
                        model=f"{self.provider}/{self.model}",
                        api_key=self.api_keys_rotator.current,
                        base_url=self.base_url,
                        messages=messages,
                        **kwargs
                    )
                    break
                except RateLimitError as e:
                    last_error = e
                    self.api_keys_rotator.next()
            else:
                raise last_error

            content = resp.choices[0].message.content
            parts = content.split("</think>", maxsplit=1)
            if len(parts) == 1:
                content = parts[0]
            else:
                content = parts[1]

            return content
        except Exception as e:
            raise e
