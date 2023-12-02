import re
from itertools import chain
from typing import List, Self, Tuple, TypeVar, Union

from modules.sd_hijack_clip import (
    FrozenCLIPEmbedderWithCustomWordsBase,
    PromptChunk,
)
from modules.sd_hijack_open_clip import FrozenOpenCLIPEmbedderWithCustomWords
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
        clip: Union[
            FrozenCLIPEmbedderWithCustomWordsBase,
            FrozenOpenCLIPEmbedderWithCustomWords,
        ],
        text: str,
    ):
        use_old = opts.use_old_emphasis_implementation
        assert not use_old, "use_old_emphasis_implementation is not supported"

        self.clip = clip
        self.id_start = clip.id_start
        self.id_end = clip.id_end
        self.is_open_clip = (
            True
            if isinstance(clip, FrozenOpenCLIPEmbedderWithCustomWords)
            else False
        )
        self.used_custom_terms = []
        self.hijack_comments = []

        chunks, token_count = self.tokenize_line(text)

        self.token_count = token_count
        self.fixes = list(chain.from_iterable(chunk.fixes for chunk in chunks))
        self.context_size = calc_context_size(token_count)

        tokens = list(chain.from_iterable(chunk.tokens for chunk in chunks))
        multipliers = list(
            chain.from_iterable(chunk.multipliers for chunk in chunks)
        )

        self.tokens = []
        self.multipliers = []
        for i in range(self.context_size // 77):
            self.tokens.extend(
                [self.id_start] + tokens[i * 75 : i * 75 + 75] + [self.id_end]
            )
            self.multipliers.extend(
                [1.0] + multipliers[i * 75 : i * 75 + 75] + [1.0]
            )

    def create(self: Self, text: str) -> TPromptAnalyzer:
        return PromptAnalyzer(self.clip, text)

    def tokenize_line(self, line) -> Tuple[PromptChunk, int]:
        chunks, token_count = self.clip.tokenize_line(line)
        return chunks, token_count

    def tokenize(self, prompts):
        return self.clip.tokenize(prompts)

    def process_text(self, texts: List[str]):
        (
            batch_multipliers,
            remade_batch_tokens,
            used_custom_terms,
            hijack_comments,
            hijack_fixes,
            token_count,
        ) = self.clip.process_text(texts)
        return (
            batch_multipliers,
            remade_batch_tokens,
            used_custom_terms,
            hijack_comments,
            hijack_fixes,
            token_count,
        )

    def encode(self, text: str):
        return self.clip.tokenize([text])[0]

    def calc_word_indecies(
        self, word: str, limit: int = -1, start_pos=0
    ) -> Tuple[List[int], int]:
        word = word.lower()
        merge_idxs = []

        tokens = self.tokens
        needles = self.encode(word)

        limit_count = 0
        current_pos = 0
        for i, token in enumerate(tokens):
            current_pos = i
            if i < start_pos:
                continue

            if needles[0] == token and len(needles) > 1:
                next = i + 1
                success = True
                for needle in needles[1:]:
                    if next >= len(tokens) or needle != tokens[next]:
                        success = False
                        break
                    next += 1

                # append consecutive indexes if all pass
                if success:
                    merge_idxs.extend(list(range(i, next)))
                    if limit > 0:
                        limit_count += 1
                        if limit_count >= limit:
                            break

            elif needles[0] == token:
                merge_idxs.append(i)
                if limit > 0:
                    limit_count += 1
                    if limit_count >= limit:
                        break

        return merge_idxs, current_pos
