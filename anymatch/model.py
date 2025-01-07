import torch
from transformers import T5Tokenizer, GPT2Tokenizer, AutoModelForSeq2SeqLM, GPT2ForSequenceClassification, \
    AutoTokenizer, LlamaForSequenceClassification

hf_token = 'your_hf_token'


class Matcher:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.print_model_info()

    def print_model_info(self):
        print(f"trainable params: {self.model.num_parameters()}", flush=True)


class T5Matcher(Matcher):
    def __init__(self, base_model: str = 't5-base'):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        self.tokenizer = T5Tokenizer.from_pretrained(base_model)
        super().__init__(self.model, self.tokenizer)


class GPTMatcher(Matcher):
    def __init__(self, base_model: str = 'gpt2'):
        self.model = GPT2ForSequenceClassification.from_pretrained(base_model)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model)
        super().__init__(self.model, self.tokenizer)


class LlamaMatcher(Matcher):
    def __init__(self, base_model: str = "meta-llama/Llama-3.2-1B"):
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token=hf_token)
        self.model = LlamaForSequenceClassification.from_pretrained("meta-llama/Llama-3.2-1B",
                     num_labels=2, token=hf_token)
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

        super().__init__(self.model, self.tokenizer)


def load_model(base_model):
    if 't5' in base_model:
        model = T5Matcher(base_model)
    elif 'gpt' in base_model:
        model = GPTMatcher(base_model)
    elif 'llama' in base_model:
        model = LlamaMatcher(base_model)
    else:
        raise ValueError('Model not found.')
    return model.model, model.tokenizer
