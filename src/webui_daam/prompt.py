import re
from typing import List, Tuple, TypeVar, Union

from modules.sd_hijack_clip import (
    FrozenCLIPEmbedderWithCustomWordsBase,
    PromptChunk,
)
from modules.sd_hijack_open_clip import (
    FrozenOpenCLIPEmbedderWithCustomWords,
)
from modules.shared import opts


def calc_context_size(token_length: int):
    len_check = 0 if (token_length - 1) < 0 else token_length - 1
    return ((int)(len_check // 75) + 1) * 77


Prompts = TypeVar("Prompts", List[str], str)


def escape_prompt(prompt: Prompts) -> Prompts:
    if isinstance(prompt, str):
        prompt = prompt.lower()
        prompt = re.sub(r"[\(\)\[\]]", "", prompt)
        prompt = re.sub(r":\d+\.*\d*", "", prompt)
        return prompt
    elif isinstance(prompt, list):
        prompt_new = []
        for i in range(len(prompt)):
            prompt_new.append(escape_prompt(prompt[i]))
        return prompt_new


TPromptAnalyzer = TypeVar("TPromptAnalyzer", bound="PromptAnalyzer")


class PromptAnalyzer:
    def __init__(
        self,
        embedders: List[
            Union[
                FrozenCLIPEmbedderWithCustomWordsBase,
                FrozenOpenCLIPEmbedderWithCustomWords,
            ]
        ],
        text: str,
    ):
        assert (
            not opts.use_old_emphasis_implementation
        ), "use_old_emphasis_implementation is not supported"

        self.embedders = embedders

        _chunks, token_count = self.tokenize_line(text)
        self.context_size = calc_context_size(token_count)
        self.token_count = token_count

    def create(self, text: str) -> TPromptAnalyzer:
        return PromptAnalyzer(self.conditioner, text)

    def tokenize_line(self, line) -> Tuple[PromptChunk, int]:
        for embedder in [
            embedder
            for embedder in self.embedders
            if hasattr(embedder, "tokenize_line")
        ]:
            return embedder.tokenize_line(line)

    def tokenize(self, texts):
        for embedder in [
            embedder
            for embedder in self.embedders
            if hasattr(embedder, "tokenize")
        ]:
            return embedder.tokenize(texts)

    def process_text(self, texts: List[str]):
        for embedder in [
            embedder
            for embedder in self.embedders
            if hasattr(embedder, "process_texts")
        ]:
            return embedder.process_texts(texts)

    def encode(self, text: str):
        return self.tokenize([text])[0]
