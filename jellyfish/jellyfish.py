import argparse
import random
import math
import time
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")


def generate_jellyfish_prompts(dataset, mode='no_columns', seed=42):
    random.seed(seed)
    data_df = pd.read_csv(f'../data/processed/{dataset}/test.csv')
    data_df = data_df.fillna('nan')
    cols = [col[:-2] for col in data_df.columns if col.endswith('_l')]
    random.shuffle(cols)
    l_cols = [f'{col}_l' for col in cols]
    r_cols = [f'{col}_r' for col in cols]

    prompt = '''You are an AI assistant that follows instruction extremely well. Help as much as you can.\n\n### Instruction:\n\nYou are tasked with determining whether two records listed below are the same based on the information provided.\n{}\nNote: Missing values (N/A or \"nan\") should not be used as a basis for your decision.\nRecord A: [{}]\nRecord B: [{}]\nAre record A and record B the same entity? Choose your answer from: [Yes, No].\n\n### Response:\n\n'''
    if mode == 'no_columns':
        attribute_text = 'Carefully compare the attributes of each record before making your decision.'
        data_df['textA'] = data_df.apply(lambda x: ', '.join([str(x[c]) for c in l_cols]), axis=1)
        data_df['textB'] = data_df.apply(lambda x: ', '.join([str(x[c]) for c in r_cols]), axis=1)
        prompts = [prompt.format(attribute_text, data_df.iloc[i]['textA'], data_df.iloc[i]['textB']) for i in
                   range(len(data_df))]
    elif mode == 'with_columns':
        attribute_text = 'Carefully compare the ' + 'attribute {}, ' * (len(cols)-1) + 'attribute {}' + 'each record before making your decision.'
        attribute_text = attribute_text.format(*range(1, len(cols)+1))
        entity_text = 'attribute {}: {}, ' * (len(cols)-1) + 'attribute {}: {}'
        cross_mix = lambda a, b: [item for tup in zip(a, b) for item in tup]
        data_df['textA'] = data_df.apply(lambda x: entity_text.format(*cross_mix(range(1, len(cols)+1), x[l_cols])), axis=1)
        data_df['textB'] = data_df.apply(lambda x: entity_text.format(*cross_mix(range(1, len(cols)+1), x[r_cols])), axis=1)
        prompts = [prompt.format(attribute_text, data_df.iloc[i]['textA'], data_df.iloc[i]['textB']) for i in range(len(data_df))]
    labels = data_df['label'].tolist()
    return prompts, labels


@torch.no_grad()
def jelly_inference(tokenizer, model, prompts, batch_size=16):
    generation_config = GenerationConfig(do_samples=True, temperature=0.35, top_p=0.9, pad_token_id=tokenizer.eos_token_id,)

    num_batches = math.ceil(len(prompts)/batch_size)
    outputs = []
    for i in range(num_batches):
        batch_prompts = prompts[i*batch_size:(i+1)*batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=350).to("cuda:0")
        generations = model.generate(
            input_ids=inputs['input_ids'],
            generation_config=generation_config,
            attention_mask=inputs["attention_mask"],
            return_dict_in_generate=True,
            output_scores=False,
            max_new_tokens=15,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15, )

        batch_outputs = tokenizer.batch_decode(generations["sequences"][:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        outputs.extend(batch_outputs)

    predictions = [1 if 'yes' in output.lower() else 0 for output in outputs]
    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The jellyfish inference experiment.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # use the model for inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForCausalLM.from_pretrained("NECOUDBFM/Jellyfish-13B",
                                                 torch_dtype=torch.float16,
                                                 device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained("NECOUDBFM/Jellyfish-13B")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    seed = args.seed
    datasets = ['abt', 'amgo', 'beer', 'dbac', 'dbgo', 'foza', 'itam', 'roim', 'waam', 'wdc', 'zoye']
    modes = ['no_columns', 'with_columns']
    for mode in modes:
        for dataset in datasets:
            prompts, labels = generate_jellyfish_prompts(dataset, mode, seed)
            start_time = time.time()
            batch_size = 16
            try:
                predictions = jelly_inference(tokenizer, model, prompts)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"CUDA OOM error: {e}. Retrying with smaller batch size.")
                    torch.cuda.empty_cache()
                    if batch_size > 1:
                        batch_size = batch_size // 2
                        predictions = jelly_inference(tokenizer, model, prompts, batch_size)
                    else:
                        print(f"Failed to run inference on {dataset}.")
                        continue
                else:
                    raise e

            end_time = time.time()
            f1 = f1_score(labels, predictions)
            acc = accuracy_score(labels, predictions)
            prec = precision_score(labels, predictions)
            rec = recall_score(labels, predictions)
            print(f'The prdictive quality on {dataset}: F1: {f1}, Acc: {acc}, Prec: {prec}, Rec: {rec}')
            print(f'Time taken: {end_time - start_time} seconds')
