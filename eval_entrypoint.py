import argparse
import subprocess
import sys


def run(cmd):
    print('Running:', ' '.join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command {' '.join(cmd)} failed with code {result.returncode}")


def eval_zeroer(dataset, seed):
    run(['python', 'zeroer/zeroer.py', dataset, '--seed', str(seed)])


def eval_ditto(dataset, seed):
    run([
        'python', 'ditto/train_ditto.py',
        '--task', f'loo-{dataset}-{seed}',
        '--batch_size', '64',
        '--max_len', '64',
        '--lr', '3e-5',
        '--n_epochs', '40',
        '--lm', 'bert',
        '--fp16',
        '--da', 'del',
        '--summarize'
    ])


def eval_unicorn(dataset, seed):
    run([
        'python', 'unicorn/main-zero-ins.py',
        '--pretrain',
        '--model', 'deberta_base',
        '--loo', dataset,
        '--seed', str(seed)
    ])


def eval_anymatch(dataset, seed, base_model):
    run([
        'python', 'anymatch/loo.py',
        '--leaved_dataset_name', dataset,
        '--base_model', base_model,
        '--seed', str(seed),
        '--tbs', '64',
        '--epochs', '25',
        '--patience_start', '10',
        '--patience', '6'
    ])


def eval_matchgpt(dataset, model):
    run([
        'python', 'matchgpt/matchgpt.py',
        '--model', model,
        '--dataset', dataset
    ])


def eval_jellyfish(seed):
    run(['python', 'jellyfish/jellyfish.py', '--seed', str(seed)])


def eval_throughput(model_name):
    run(['python', 'throughput.py', '--model_name', model_name])


def main():
    parser = argparse.ArgumentParser(description='Unified evaluation entrypoint')
    parser.add_argument('--method', required=True,
                        choices=['zeroer', 'ditto', 'unicorn', 'anymatch',
                                 'matchgpt', 'jellyfish', 'throughput'])
    parser.add_argument('--dataset', help='Dataset name for the evaluation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--base_model', help='Base model for AnyMatch')
    parser.add_argument('--model', help='Model for MatchGPT or throughput')
    args = parser.parse_args()

    if args.method == 'zeroer':
        if not args.dataset:
            parser.error('--dataset is required for zeroer')
        eval_zeroer(args.dataset, args.seed)
    elif args.method == 'ditto':
        if not args.dataset:
            parser.error('--dataset is required for ditto')
        eval_ditto(args.dataset, args.seed)
    elif args.method == 'unicorn':
        if not args.dataset:
            parser.error('--dataset is required for unicorn')
        eval_unicorn(args.dataset, args.seed)
    elif args.method == 'anymatch':
        if not args.dataset:
            parser.error('--dataset is required for anymatch')
        base = args.base_model or 'llama3'
        eval_anymatch(args.dataset, args.seed, base)
    elif args.method == 'matchgpt':
        if not args.dataset:
            parser.error('--dataset is required for matchgpt')
        model = args.model or 'mixtral'
        eval_matchgpt(args.dataset, model)
    elif args.method == 'jellyfish':
        eval_jellyfish(args.seed)
    elif args.method == 'throughput':
        model = args.model or 'anymatch-llama3'
        eval_throughput(model)
    else:
        parser.error('Unsupported method')


if __name__ == '__main__':
    main()
