
# Emulating hugging face tokenizer
class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, prompt):
        return self.tokenizer(prompt)
