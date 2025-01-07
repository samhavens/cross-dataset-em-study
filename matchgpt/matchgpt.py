import argparse
import math
import random
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")

hf_key = 'your_hf_key'

def get_prompts(model_name, dataset, seed=42):
    if model_name == 'mixtral':
        prompt = '''[INST]Do the two entity descriptions refer to the same real-world entity? Answer with 'Yes' if they do and 'No' if they do not.\nEntity 1: {}\nEntity 2: {}[/INST]'''
    elif model_name == 'solar':
        prompt = '''### User:\n Do the two entity descriptions refer to the same real-world entity? Answer with 'Yes' if they do and 'No' if they do not.\nEntity 1: {}\nEntity 2: {}\n\n### Assistant:\n'''
    elif model_name == 'beluga':
        prompt = '''### System:\nYou are Stable Beluga, an AI that follows instructions extremely well.\n\n### User: Do the two entity descriptions refer to the same real-world entity? Answer with 'Yes' if they do and 'No' if they do not.\nEntity 1: {}\nEntity 2: {}\n\n### Assistant:\n'''
    else:
        raise ValueError(f"Model {model_name} not supported.")

    random.seed(seed)
    data_df = pd.read_csv(f'../data/processed/{dataset}/test.csv')
    data_df = data_df.fillna('nan')
    cols = [col[:-2] for col in data_df.columns if col.endswith('_l')]
    random.shuffle(cols)
    l_cols = [f'{col}_l' for col in cols]
    r_cols = [f'{col}_r' for col in cols]

    data_df['text'] = data_df.apply(lambda x: prompt.format('\t'.join([str(x[c]) for c in l_cols]),
                                                                  '\t'.join([str(x[c]) for c in r_cols])), axis=1)
    return data_df['text'].tolist(), data_df['label'].tolist()

def get_model(model_name):
    if model_name == 'mixtral':
        model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    elif model_name == 'solar':
        model_id = "upstage/Llama-2-70b-instruct-v2"
    elif model_name == 'beluga':
        model_id = "stabilityai/StableBeluga2"
    else:
        raise ValueError(f"Model {model_name} not supported.")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir='your_cache_dir',
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        load_in_8bit=True,
        use_auth_token=hf_key
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, use_auth_token=hf_key)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


@torch.no_grad()
def matchgpt_inference(tokenizer, model, prompts, batch_size=8):
    num_batches = math.ceil(len(prompts)/batch_size)
    outputs = []
    for i in range(num_batches):
        batch_prompts = prompts[i*batch_size:(i+1)*batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=300).to("cuda:0")
        generations = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs["attention_mask"],
            return_dict_in_generate=True,
            output_scores=False,
            max_new_tokens=15,
            pad_token_id=tokenizer.eos_token_id)

        batch_outputs = tokenizer.batch_decode(generations["sequences"][:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        outputs.extend(batch_outputs)

    predictions = [1 if 'yes' in output.lower() else 0 for output in outputs]
    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The jellyfish inference experiment.')
    parser.add_argument('--model', type=str, default='mixtral', choices=['mixtral', 'solar', 'beluga'])
    parser.add_argument('--dataset', type=str, default='abt')
    args = parser.parse_args()

    model_name = args.model
    seeds = [42, 44, 46, 48, 50]
    # seeds = [42]
    dataset = args.dataset
    model, tokenizer = get_model(model_name)

    for seed in seeds:
        prompts, labels = get_prompts(model_name, dataset, seed)
        start_time = time.time()
        batch_size = 16
        while True:
            try:
                predictions = matchgpt_inference(tokenizer, model, prompts)
                break
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"CUDA OOM error: {e}. Retrying with smaller batch size.")
                    torch.cuda.empty_cache()
                    if batch_size > 1:
                        batch_size = batch_size // 2
                        predictions = matchgpt_inference(tokenizer, model, prompts, batch_size)
                    else:
                        print(f"Failed to run inference on {dataset}.")
                        break
                else:
                    raise e

        end_time = time.time()
        f1 = f1_score(labels, predictions)
        acc = accuracy_score(labels, predictions)
        prec = precision_score(labels, predictions)
        rec = recall_score(labels, predictions)
        print(f'The prdictive quality on {dataset}: F1: {f1}, Acc: {acc}, Prec: {prec}, Rec: {rec}')
        print(f'Time taken: {end_time - start_time} seconds')


