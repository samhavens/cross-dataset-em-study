import argparse
import copy
import os
import pandas as pd

from utils.data_utils import read_multi_row_data, read_multi_attr_data, read_single_row_data
from utils.train_eval import train, inference
from data import T5Dataset, GPTDataset, LlamaDataset
from model import load_model


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_loo_dirs(dataset_name):
    dataset_names = ['abt', 'amgo', 'beer', 'dbac', 'dbgo', 'foza', 'itam', 'roim', 'waam', 'wdc', 'zoye']
    loo_dataset_names = [dn for dn in dataset_names if dn != dataset_name]
    loo_dataset_dirs = [f'../data/processed/{dn}' for dn in loo_dataset_names]
    return loo_dataset_dirs


parser = argparse.ArgumentParser(description='The fast leave one out experiment.')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--tbs', type=int, default=32)
parser.add_argument('--base_model', type=str, default='llama3')
parser.add_argument('--leaved_dataset_name', type=str, default='wdc')
parser.add_argument('--serialization_mode', type=str, default='mode1')
parser.add_argument('--row_sample_func', type=str, default='automl_filter')
parser.add_argument('--train_data', type=str, default='attr+row', choices=['row', 'attr+row', 'attr-row'])
parser.add_argument('--patience_start', type=int, default=-1)
parser.add_argument('--patience', type=int, default=3)
args = parser.parse_args()

seed = args.seed
epochs = args.epochs
base_model = args.base_model
leaved_dataset_name = args.leaved_dataset_name
serialization_mode = args.serialization_mode
row_sample_func = args.row_sample_func
train_data = args.train_data
patience_start = args.patience_start
patience = args.patience

model, tokenizer = load_model(base_model)
dataset_dirs = get_loo_dirs(leaved_dataset_name)

if base_model == 't5-base':
    lr = 1e-4
    DatasetClass = T5Dataset
elif base_model == 'gpt2':
    lr = 2e-5
    DatasetClass = GPTDataset
elif base_model == 'llama3':
    lr = 1e-6
    DatasetClass = LlamaDataset
else:
    raise ValueError('The base model is currently not supported.')

tbs = args.tbs

print('-----' * 10)
print(f'Experiment to leave the {leaved_dataset_name} dataset out with {train_data} as training data.', flush=True)
if train_data == 'attr-row':
    print('The model firstly be pre-trained on the attribute pairs to get familiar with the EM task.', flush=True)
    train_attr_df, valid_attr_df, _ = read_multi_attr_data(dataset_dirs, serialization_mode)
    train_attr_d = DatasetClass(tokenizer, train_attr_df, max_len=350)
    valid_attr_d = DatasetClass(tokenizer, valid_attr_df, max_len=350)
    best_model = train(tokenizer, model, train_attr_d, valid_attr_d, epochs=5, lr=lr, seed=seed, patient=True,
                       save_model=False, save_freq=50, train_batch_size=tbs, valid_batch_size=64, save_model_path='',
                       save_result_prefix='', patience=2, patience_start=-1, base_model=base_model)
    model = copy.deepcopy(best_model)
    print('The pre-training phase is finished.', flush=True)
    train_df, valid_df, _ = read_multi_row_data(dataset_dirs, serialization_mode, row_sample_func, seed=seed)

elif train_data == 'row':
    print('The model will be trained on the row level data.', flush=True)
    train_df, valid_df, _ = read_multi_row_data(dataset_dirs, serialization_mode, row_sample_func, seed=seed)

elif train_data == 'attr+row':
    print('The model will be trained on the mixture of attribute and row level data.', flush=True)
    train_attr_df, _, _ = read_multi_attr_data(dataset_dirs, serialization_mode)
    train_row_df, valid_row_df, _ = read_multi_row_data(dataset_dirs, serialization_mode, row_sample_func, seed=seed)
    train_df = pd.concat([train_attr_df, train_row_df], ignore_index=True).drop_duplicates().reset_index(drop=True)
    valid_df = valid_row_df

train_d = DatasetClass(tokenizer, train_df, max_len=350)
valid_d = DatasetClass(tokenizer, valid_df, max_len=350)

print('The training phase starts from here.', flush=True)
print(f'The size of the training and validation datasets are: {len(train_d)}, {len(valid_d)}', flush=True)
print(f'Here is the configuration for the experiment:\n'
      f'\tseed: {seed}\tbase_model: {base_model}\tdataset_name: {leaved_dataset_name}\tmode: {serialization_mode} '
      f'\tmax_len: {350}\tlr: {lr}\tbatch_size: {tbs}\tpatience: {patience}\tp_start: {patience_start}', flush=True)

saved_model_path = f'saved_models/loo_{leaved_dataset_name}_{base_model}'
best_model = train(tokenizer, model, train_d, valid_d, epochs=epochs, lr=lr, seed=seed, patient=True, save_model=False,
                   save_freq=50, train_batch_size=tbs, valid_batch_size=32, save_model_path=saved_model_path,
                   save_result_prefix='', patience=patience, patience_start=patience_start, base_model=base_model)
print('The training phase is finished.', flush=True)

# best_model.module.save_pretrained(f"models/{base_model}")

print('Start the evaluation phase.')
_, _, test_df = read_single_row_data(f'../data/processed/{leaved_dataset_name}', serialization_mode, print_info=False, seed=seed)
test_d = DatasetClass(tokenizer, test_df, max_len=350)
test_f1, test_acc, test_prec, test_rec = inference(tokenizer, best_model, test_d, batch_size=32, base_model=base_model)
print(f'Test acc, f1, prec, recall for {leaved_dataset_name} is {test_acc * 100:.2f}, {test_f1 * 100:.2f}, '
      f'{test_prec * 100:.2f}, {test_rec * 100:.2f}', flush=True)
print('Evaluation finished.', flush=True)

print('-----' * 10)

