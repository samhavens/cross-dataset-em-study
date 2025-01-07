import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer

from anymatch.utils.data_utils import read_single_row_data

import pandas as pd
import torch
from torch.utils.benchmark import Timer

from jellyfish import jellyfish as get_openmodel_prompts


def prepare_prompts(model_name):
    if model_name in ['ditto', 'anymatch-gpt2', 'anymatch-t5', 'anymatch-llama3']:
        return read_single_row_data('data/prepared/dbgo/train.csv')
    elif model_name in ['jellyfish', 'mixtral', 'beluga', 'solar']:
        return get_openmodel_prompts('dbgo')[0]


def print_gpu_memory_usage():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} | Name: {torch.cuda.get_device_name(i)}")
        print(f"     | Allocated: {torch.cuda.memory_allocated(i) / (1024 ** 3):.3f} GB")
        print(f"     | Cached: {torch.cuda.memory_reserved(i) / (1024 ** 3):.3f} GB")


def benchmark_inference(model, tokenizer, dataset, initial_batch_size=4, max_batch_size=16384):
    """
    Benchmarks the inference time of a model with the largest possible batch size that fits in memory,
    and counts the number of tokens processed.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Helper function to test if a batch size fits in memory
    def can_fit_in_memory(batch_size):
        try:
            inputs = tokenizer(dataset[:batch_size], return_tensors="pt", truncation=True, max_length=350, padding=True)
            inputs['labels'] = torch.tensor([[2000, 1]] * batch_size)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            with torch.no_grad():
                model(**inputs)
            return True
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                return False
            else:
                raise e

    # Determine the largest batch size that fits in memory
    batch_size = initial_batch_size
    while batch_size <= max_batch_size:
        if can_fit_in_memory(batch_size):
            largest_batch_size = batch_size
            batch_size *= 2
            print(f"Batch size {largest_batch_size} fits in memory.")
        else:
            break

    # Prepare the inputs with the determined largest batch size
    inputs = tokenizer(dataset[:largest_batch_size], return_tensors="pt", truncation=True, max_length=350, padding=True)
    inputs['labels'] = torch.tensor([[2000, 1]] * largest_batch_size)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Count the total number of tokens processed
    total_tokens = sum(len(token_ids) for token_ids in inputs['input_ids'])

    def inference():
        with torch.no_grad():
            model(**inputs)

    # Benchmark the inference time
    t = Timer(
        stmt='inference()', globals=locals()
    )

    result = t.timeit(100)

    # Return the results
    return {
        "batch_size": largest_batch_size,
        "total_time": result.mean * 100,  # Total time for 100 runs
        "avg_time_per_inference": result.mean,  # Average time per inference
        "tokens_processed": total_tokens,
        "throughput_of_records": largest_batch_size / result.mean,  # Throughput inferences per second
        "throughput_of_tokens": total_tokens / result.mean,  # Throughput tokens per second
    }


parser = argparse.ArgumentParser(description='The inference throughput experiment.')
parser.add_argument('--model_name', type=str, default='anymatch-llama3')
args = parser.parse_args()

model_name = args.model_name
if model_name == 'ditto':
    from ditto.ditto_light.ditto import DittoModel
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = DittoModel(lm='bert')
    model.load_state_dict(torch.load('saved_models/ditto/dbgo'))
elif model_name == 'anymatch-gpt2':
    from anymatch.model import GPTMatcher
    model, tokenizer = GPTMatcher('gpt2')
    model.load_state_dict(torch.load('saved_models/anymatch/gpt2/dbgo'))
elif model_name == 'anymatch-t5':
    from anymatch.model import T5Matcher
    model, tokenizer = T5Matcher('t5-base')
    model.load_state_dict(torch.load('saved_models/anymatch/t5/dbgo'))
elif model_name == 'anymatch-llama3':
    from anymatch.model import LlamaMatcher
    model, tokenizer = LlamaMatcher('meta-llama/Llama-3.2-1B')
    model.load_state_dict(torch.load('saved_models/anymatch/llama3/dbgo'))
elif model_name == 'unicorn':
    print('the inference of unicorn is a very different, a separate script is needed')
elif model_name == 'jellyfish':
    model = AutoModelForCausalLM.from_pretrained("NECOUDBFM/Jellyfish-13B", torch_dtype=torch.float16, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained("NECOUDBFM/Jellyfish-13B")
elif model_name == 'mixtral':
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", use_fast=False, token="your_token")
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        device_map="auto",
        torch_dtype=torch.float16,
        token="your_token"
    )
elif model_name == 'beluga':
    tokenizer = AutoTokenizer.from_pretrained("stabilityai/StableBeluga2", use_fast=False,
                                              token="your_token")
    model = AutoModelForCausalLM.from_pretrained(
        "stabilityai/StableBeluga2",
        device_map="auto",
        torch_dtype=torch.float16,
        token="your_token"
    )
elif model_name == 'solar':
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-70B-Instruct", use_fast=False,
                                              token="your_token")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16,
        token="your_token"
    )
else:
    raise ValueError('Model not found.')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print('Inference benchmarking on {} model...'.format(model_name))
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print_gpu_memory_usage()
dataset = prepare_prompts(model_name)
benchmark_results = benchmark_inference(model, tokenizer, dataset)
print(benchmark_results)
print('Benchmarking finished.')
