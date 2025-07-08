import argparse
import subprocess

# Default list of benchmark datasets used across the repository
DATASETS = ["abt", "amgo", "beer", "dbac", "dbgo", "foza", "itam", "roim", "waam", "wdc", "zoye"]


def run(cmd):
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command {' '.join(cmd)} failed with code {result.returncode}")


def eval_zeroer(dataset, seed):
    run(["python", "zeroer/zeroer.py", dataset, "--seed", str(seed)])


def eval_ditto(dataset, seed):
    run(
        [
            "python",
            "ditto/train_ditto.py",
            "--task",
            f"loo-{dataset}-{seed}",
            "--batch_size",
            "64",
            "--max_len",
            "64",
            "--lr",
            "3e-5",
            "--n_epochs",
            "40",
            "--lm",
            "bert",
            "--fp16",
            "--da",
            "del",
            "--summarize",
        ]
    )


def eval_unicorn(dataset, seed):
    run(
        [
            "python",
            "unicorn/main-zero-ins.py",
            "--pretrain",
            "--model",
            "deberta_base",
            "--loo",
            dataset,
            "--seed",
            str(seed),
        ]
    )


def eval_anymatch(dataset, seed, base_model):
    run(
        [
            "python",
            "anymatch/loo.py",
            "--leaved_dataset_name",
            dataset,
            "--base_model",
            base_model,
            "--seed",
            str(seed),
            "--tbs",
            "64",
            "--epochs",
            "25",
            "--patience_start",
            "10",
            "--patience",
            "6",
        ]
    )


def eval_matchgpt(dataset, model):
    run(["python", "matchgpt/matchgpt.py", "--model", model, "--dataset", dataset])


def eval_jellyfish(seed):
    run(["python", "jellyfish/jellyfish.py", "--seed", str(seed)])


def eval_throughput(model_name):
    run(["python", "throughput.py", "--model_name", model_name])


def eval_random_clustering(dataset, seed):
    run(["python", "random_clustering.py", dataset, "--seed", str(seed)])


def eval_custom(dataset, seed, script):
    cmd = ["python", script, dataset]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    run(cmd)


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation entrypoint")
    parser.add_argument(
        "--method",
        required=True,
        choices=[
            "zeroer",
            "ditto",
            "unicorn",
            "anymatch",
            "matchgpt",
            "jellyfish",
            "throughput",
            "random_clustering",
            "custom",
        ],
    )
    parser.add_argument("--dataset", help="Dataset name for the evaluation")
    parser.add_argument("--all", action="store_true", help="Run on all benchmark datasets")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="List of seeds to run")
    parser.add_argument("--base_model", help="Base model for AnyMatch")
    parser.add_argument("--model", help="Model for MatchGPT or throughput")
    parser.add_argument("--script", help="Custom evaluation script when using method=custom")
    args = parser.parse_args()

    if args.all:
        datasets = DATASETS
    else:
        if not args.dataset:
            parser.error("--dataset is required unless --all is set")
        datasets = [args.dataset]

    for dataset in datasets:
        for seed in args.seeds:
            if args.method == "zeroer":
                eval_zeroer(dataset, seed)
            elif args.method == "ditto":
                eval_ditto(dataset, seed)
            elif args.method == "unicorn":
                eval_unicorn(dataset, seed)
            elif args.method == "anymatch":
                base = args.base_model or "llama3"
                eval_anymatch(dataset, seed, base)
            elif args.method == "matchgpt":
                model = args.model or "mixtral"
                eval_matchgpt(dataset, model)
            elif args.method == "jellyfish":
                eval_jellyfish(seed)
            elif args.method == "throughput":
                model = args.model or "anymatch-llama3"
                eval_throughput(model)
            elif args.method == "random_clustering":
                eval_random_clustering(dataset, seed)
            elif args.method == "custom":
                if not args.script:
                    parser.error("--script is required for custom method")
                eval_custom(dataset, seed, args.script)
            else:
                parser.error("Unsupported method")


if __name__ == "__main__":
    main()
