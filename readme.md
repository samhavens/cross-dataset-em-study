# Cross-Dataset-EM-Study

<div style="display: flex; justify-content: space-between; gap: 20px;">
  <img src="results/lodo.png" style="width: 48%; height: auto;" alt="Lodo Image">
  <img src="results/f1-vs-cost.png" style="width: 48%; height: auto;" alt="F1 vs Cost Image">
</div>

This repository is the codebase of experiments and artificats of paper "[Experiments & Analysis] A Deep Dive Into Cross-Dataset Entity Matching with Large and Small Language Models". 

It includes the basic components and configurations for reproducing the evaluations described in the paper. For detailed implementations of each method, please refer to the original implementations linked here.

## Quick Start

### Environment Setup

To set up the environment and install all dependencies:

```bash
# Clone the repository
git clone https://github.com/samhavens/cross-dataset-em-study.git
cd cross-dataset-em-study

# Set up environment (works with both uv and pip)
./setup.sh

# Activate the environment
source .venv/bin/activate

# Run a quick benchmark (works in mock mode without API keys)
./bin/quick_llm_bench.sh

# Or test the system
python test_llm_bench.py

# Test the new LLM hybrid entity matching (mock mode)
python llm_em_hybrid.py --dataset beer --limit 20 --candidate-ratio 0.05

# Test dataset analysis tools
python tools/dataset_info.py --dataset beer

# Build LLM keys for a dataset (with progress bar and checkpointing)
python tools/build_llm_keys.py --dataset beer --model gpt-4.1-nano
```

### Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

The project requires:
- **Core**: pandas, numpy, scikit-learn
- **LLM**: dspy-ai, tiktoken, transformers, torch
- **ML**: autogluon, sentence-transformers
- **String matching**: jellyfish, fuzzywuzzy
- **Analysis**: jupyter, matplotlib, seaborn

See `requirements.txt` for the complete list with version constraints.

### API Keys (Optional)

The system works in mock mode by default for testing. To use real LLM APIs:

```bash
# Set your OpenAI API key (will automatically enable real API calls)
export OPENAI_API_KEY="your-key-here"
./bin/quick_llm_bench.sh
```

> **Note:** The code requires **DSPy 2.6.0** (or later). Older releases used
> the `Graph`/`Map` names; we now depend on `Composition`/`Parallel`.
> Run `pip install -U dspy-ai==2.6.0` if you hit attribute errors.

* ZeroER: https://github.com/chu-data-lab/zeroer
* Ditto: https://github.com/megagonlabs/ditto
* Unicorn: https://github.com/ruc-datalab/Unicorn
* AnyMatch: https://github.com/Jantory/anymatch
* Jellyfish: https://huggingface.co/NECOUDBFM/Jellyfish-13B
* MatchGPT: https://github.com/wbsg-uni-mannheim/MatchGPT
* FM-Data-Tasks (GPT3): "https://github.com/HazyResearch/fm_data_tasks"
* TableGPT: https://github.com/microsoft/Table-GPT



## Dataset
We use ten recognized benchmark datasets from the Magellan repository (the first eight are widely used in existing studies), 
along with the WDC dataset, which is a recent addition from the e-commerce data. 
For detailed source information about these datasets, please visit the links provided below:

|Abbr.| Dataset        |                                                                                              Link | 
|:---|:---------------|--------------------------------------------------------------------------------------------------:|
|wdc| wdc             |                                    [wdc](https://webdatacommons.org/largescaleproductcorpus/v2/ )  | 
|abt| abt_buy         |                  [magellan](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md  )  | 
|amgo| amazon_google  |                  [magellan](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md  ) | 
|beer| beer           |                  [magellan](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md  ) | 
|dbac| dblp_acm       |                  [magellan](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md  ) | 
|dbgo| dblp_scholar   |                  [magellan](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md  ) | 
|foza| fodors_zagat   |                  [magellan](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md  ) | 
|itam| itunes_amazon  |                  [magellan](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md  ) | 
|waam| walmart_amazon |                  [magellan](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md  ) | 
|roim| rottentomato_imdb| [magellan data](https://sites.google.com/site/anhaidgroup/useful-stuff/the-magellan-data-repository)|
|zoye| zomato_yelp| [magellan data](https://sites.google.com/site/anhaidgroup/useful-stuff/the-magellan-data-repository)|\


The training and validation sets for each dataset originate from their respective sources. However, the test set is down-sampled to a maximum of 1,250 samples to optimize the cost of OpenAI API calls. For all baseline comparisons, the test set remains identical, adhering to the same leave-one-dataset-out configuration.

To conserve space, only the raw datasets are included in this repository. The code for processing and formatting them is available in `data/data_preparation.ipynb`. The datasets are first processed into a base format, which is subsequently adapted to create method-specific datasets. 

Moreover, to address the data leakage issue that might happend during fine-tuning phase, we conduct a SQL analysis to validate there is zero overlapping datapoints between any pair of datasets, which can be found in `data/data_leakage.py`.


## Different cross-dataset EM matchers
We compared the efficency of the following cross-dataset matchers, from both the predictive quality and cost dimensions.

### ZeroER
The code can be found in the `zeroer` folder. We use the vanilla implementation with the application of the transitivity constraint. To run the experiments, navigate to the folder and run --
```
python zeroer.py DATASET_NAME --run_transitivity
``` 

### Ditto
The code is located in the `ditto` folder. To comply with our 'leave-one-dataset-out' strategy, the training, validation, and test data must be configured accordingly, resulting in a new config.json file in the folder. Additionally, the dataset needs to be serialized into a format that Ditto recognizes. We provide the necessary code for this in the 'ditto/data' folder. To run the experiments, navigate to the folder and run --
```
python train_ditto.py \ --task DATASET_NAME \
  --batch_size 64 --max_len 64 --lr 3e-5 \
  --n_epochs 40 --lm bert --fp16 --da del --summarize
``` 

### Unicorn
The code is located in the `unicorn` folder. To ensure the evaluation aligns with the 'leave-one-dataset-out' strategy, we adjust the 'dataprocess/dataformat.py' file, using an ordinal number to represent each dataset that is left out. To run the experiments, navigate to the folder and run --
```
python main-zero-ins.py --pretrain --model deberta_base --loo DATASET_ORDINAL
```

### AnyMatch
The code is located in the `anymatch` folder. To run the experiments, navigate to the folder and run --
```
python loo.py --leaved_dataset_name DATASET_NAME --base_model BASE_MODEL
``` 

### Jellyfish
The code is located in the `jellyfish` folder. We use the prompt suggested by the authors and download the model directly from Hugging Face for inference. A script to run the experiments is also provided
```
python jellyfish.py
```

### MatchGPT
The code is located in the matchgpt folder. Instead of using a notebook for all experiments, we adopt the same prompt system described in the original MatchGPT paper and convert everything into a Python script for improved parallel execution. For GPT models, we provide code both with and without demonstrations.
To run inference with GPT models, please use the following command:
```
python gpt.py --mid MODEL_ID --dem DEMONSTRATION_METHOD
```

To run other open models, use the following command:
```
python open_models.py --mid MODEL_ID
```

### TableGPT & GPT-3
We are unable to evaluate these two models because TableGPT is not open-sourced, and GPT-3 has been deprecated. Therefore, we include their results from the original papers for reference.

## Unified evaluation entry point
To simplify running evaluations across different methods, you can use
`eval_entrypoint.py` from the repository root. It dispatches to the
corresponding evaluation code for each matcher or a custom script.
Example usage:
```bash
python eval_entrypoint.py --method zeroer --dataset abt --seed 42
python eval_entrypoint.py --method anymatch --dataset dbgo --base_model llama3
```

For lightweight experimentation or CI checks, a toy `random_clustering` method
is provided. It predicts matches by randomly assigning records to clusters. The
script prints both the actual F1 score and the theoretical expectation based on
the number of clusters. You can run it on a single dataset or all datasets:
```bash
python eval_entrypoint.py --method random_clustering --dataset abt
python eval_entrypoint.py --method random_clustering --all
```

To benchmark a new method on all datasets at once, pass the path to the
script and use `--method custom --all`:
```bash
python eval_entrypoint.py --method custom --script my_method.py --all
```

## Inference throughput experiments
We provide the code to run the throughput experiments. To run the experiments, please use the following code:
```
python throughput.py --model_name MODEL_NAME
```

## Experimental results and analysis
The raw results for the reported numbers in Table 3 and Table 4 can be found in `results`. Moreover, a separate notebook containing all the analyses presented in the paper is available in `results/analysis.ipynb`.



#### llm mini-key blocking

* one **gpt-4.1-nano** call per canonical record produces a ≤10-token key
  (cached in `<dataset>_llmkeys.pkl`).
* at match-time a single nano call turns the left record into its key,
  we retrieve all right records whose key is exact or edit-distance ≤ 1,
  and feed only that block to the llm matcher.
* recall ≥ 99 % on abt_buy & amazon_google with ~200 candidates / row,
  cost ≈ $0.02 per 500 matches (keys amortised).

build keys once per dataset:

```bash
python tools/build_llm_keys.py --dataset abt_buy
```

## LLM Entity Matching

### Hybrid Entity Matching

We provide an efficient hybrid approach that combines trigram similarity for candidate filtering with LLM-based matching:

```bash
# Run entity matching with candidate filtering
python llm_em_hybrid.py --dataset beer --candidate-ratio 0.05 --model gpt-4.1-nano

# Alternative: specify absolute number of candidates
python llm_em_hybrid.py --dataset beer --max-candidates 150 --model gpt-4.1-nano

# Run on validation/test subset for quick testing
python llm_em_hybrid.py --dataset beer --limit 100 --candidate-ratio 0.02
```

**Key Features:**
- **Candidate filtering**: Uses trigram similarity to reduce candidates from full Table B to manageable subset
- **Token-aware**: Automatically respects 1M token context limits 
- **Cost-efficient**: ~$0.02 per 500 matches vs $2+ for naive approaches
- **Progress tracking**: Real-time progress bars and match counting

### Hyperparameter Optimization

Find optimal candidate ratios automatically using validation set sweeping:

```bash
# Sweep candidate ratios to find optimal F1
python tools/sweep_candidates.py --dataset beer --num-points 8

# Quick sweep with limited test pairs
python tools/sweep_candidates.py --dataset fodors_zagat --limit 50 --num-points 5

# Test specific dataset constraints
python tools/dataset_info.py --dataset walmart_amazon
```

The sweeping tool:
- Tests multiple candidate ratios using logarithmic spacing
- Uses validation set for hyperparameter tuning
- Estimates safe token limits automatically
- Compares results against published leaderboard thresholds
- Provides optimal configuration recommendations

### Dataset Analysis Tools

```bash
# Get dataset size and competitive thresholds
python tools/dataset_info.py --dataset beer
python tools/dataset_info.py --test-parsing  # Test all datasets

# Check what datasets are available
python tools/dataset_info.py --test-parsing
```

**Output includes:**
- Table B record counts and test pair statistics
- Average tokens per record for cost estimation
- Maximum safe candidate ratios to avoid token limits
- Competitive F1 thresholds from published leaderboards

### Alternative Key Building Methods

For large datasets, we provide optimized key building approaches:

```bash
# Method 1: Mega-prompts (1-5 API calls total)
python tools/build_llm_keys_mega.py --dataset walmart_amazon --model gpt-4.1-nano

# Method 2: Batch API (75% cost savings, 24h latency)
python tools/build_llm_keys_batch.py --dataset dblp_scholar --model gpt-4.1-nano

# Method 3: Standard batching (original, most reliable)
python tools/build_llm_keys.py --dataset beer --model gpt-4.1-nano
```

**Comparison:**
- **Standard**: 40-record batches, checkpointing, retry logic (~3 min/dataset)
- **Mega**: Uses full 1M token context, 1-5 calls total (~30 sec/dataset)
- **Batch API**: Async batch processing, 75% cheaper, 24h turnaround

## Quick Examples

### Complete Workflow Example

```bash
# 1. Set up environment
export OPENAI_API_KEY="your-key-here"

# 2. Build keys for dataset (one-time cost)
python tools/build_llm_keys.py --dataset beer --model gpt-4.1-nano

# 3. Find optimal candidate ratio
python tools/sweep_candidates.py --dataset beer --limit 50

# 4. Run final evaluation with optimal settings
python llm_em_hybrid.py --dataset beer --candidate-ratio 0.03 --model gpt-4.1-nano

# 5. Compare against leaderboard
python tools/dataset_info.py --dataset beer
```

### Cost Optimization

```bash
# Check dataset size before running
python tools/dataset_info.py --dataset walmart_amazon

# For large datasets, use batch API for key building
python tools/build_llm_keys_batch.py --dataset walmart_amazon

# Use smaller candidate ratios for cost control
python llm_em_hybrid.py --dataset walmart_amazon --candidate-ratio 0.01
```

## Advanced Experimental Features

### Structured Results and CSV Logging

Save detailed experimental results in structured formats:

```bash
# Save results to JSON and CSV
python llm_em_hybrid.py --dataset beer --candidate-ratio 0.05 \
    --output-json results.json --output-csv experiment_log.csv

# All subsequent runs append to the same CSV for experiment tracking
python llm_em_hybrid.py --dataset fodors_zagat --candidate-ratio 0.02 \
    --output-csv experiment_log.csv
```

**Structured output includes:**
- Complete hyperparameters and dataset info
- Detailed metrics (precision, recall, F1, confusion matrix)
- Cost tracking and timing information
- Reproducible experiment metadata

### Automated Hyperparameter Optimization with Test Evaluation

Run complete experiments with automatic validation→test pipeline:

```bash
# Sweep validation set, then auto-test best config on test set
python tools/sweep_candidates.py --dataset beer --auto-test \
    --output-csv experiments.csv

# Multiple datasets with experiment tracking
python tools/run_experiments.py --datasets beer fodors_zagat zomato_yelp \
    --output-csv all_experiments.csv

# Analyze all experimental results
python tools/run_experiments.py --analyze-only
```

**Auto-test pipeline:**
1. Sweeps candidate ratios on validation set
2. Identifies best hyperparameters by F1 score
3. Automatically runs best config on test set
4. Compares against competitive leaderboard thresholds
5. Saves complete experimental record

### Batch Experimental Workflows

Run comprehensive experiments across multiple datasets:

```bash
# Run experiments on all small datasets
python tools/run_experiments.py

# Custom dataset selection with debugging
python tools/run_experiments.py --datasets beer walmart_amazon \
    --limit 100 --num-points 5

# Production run with full hyperparameter search
python tools/run_experiments.py --datasets beer fodors_zagat \
    --num-points 12 --model gpt-4.1-nano
```

**Features:**
- Automatic dataset size detection and safety limits
- Progressive experiment tracking with CSV logs
- Results aggregation and competitive benchmarking
- Cost and timing analysis across experiments

### Experimental Analysis Tools

```bash
# Load and analyze experimental CSV logs
python -c "
import pandas as pd
df = pd.read_csv('experiment_results/all_results.csv')
print(f'Datasets tested: {df.dataset.nunique()}')
print(f'Best F1 scores:')
print(df.groupby('dataset')['f1'].max().sort_values(ascending=False))
"

# Compare validation vs test performance
python tools/run_experiments.py --analyze-only --results-dir experiment_results
```
