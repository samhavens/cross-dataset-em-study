# Cross-Dataset Entity Matching Pipeline

Entity Matching: Automatically find records that refer to the same real-world entity across different datasets (e.g., matching "iPhone 15" in Amazon to "Apple iPhone 15 Pro" in Best Buy).

This system uses a RAG architecture. For identifying which records in Table B (called "right") a given element of Table A (called "left") is, the system:

Does a weighted average of 3 similarities:

- Semantic: "Meaning" via embeddings
- Trigram: Character overlap
- Syntactic: String patterns and structure

Then it sends the left record and top N right records to an LLM (default: gpt-4.1-nano) asking for the index of the matching record.

## Quick Start

```bash
# Set up environment
./setup.sh
source .venv/bin/activate
export OPENAI_API_KEY="your-openai-key"

python run_complete_pipeline.py --dataset beer

# or if using TEI

python run_complete_pipeline --datasets all --embedding-base-url http://localhost:8080 --embedding-model tei --max-candidates 10
```

### Using better embeddings

You can set `--embedding-model` to any SentenceTransformers model. Alternatively, you can use any OpenAI API compatible embeddings server. I used Qwen3-0.6B via [Text Embeddings Inference](https://huggingface.co/docs/text-embeddings-inference/en/quick_tour)

```sh
model=Qwen/Qwen3-Embedding-0.6B
volume=$PWD/data

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.7 --model-id $model
```

## Architecture

  1. Dev Analysis → Analyze sample data to understand matching patterns
  2. Agentic Optimization → Claude optimizes the prompt and retrieval settings (syntactic vs semantic, # of candidates) against the dev set. There is also old code for generating heuristics. I couldn't get it to boost scores but it can cut the cost and latency down for a slight quality hit
  3. Test Evaluation → Run full test with optimized parameters
  4. Results → F1 scores, costs, and performance metrics

## How does Agentic Optimization work?

There is an MCP server which gives Claude tools to inspect the results of a run (as well as some supplementary recall analysis after the initial run). Claude can then test different weightings with the TestWeights tool and see how it impacts ordering or TableB results given a TableA record. When ready, Claude can call RunExperiment and it does a full dev set run with Claude's chosen weights and prompt. The config and results of the experiment are saved. After some rounds of iteration, Claude can decide to be done, and the best experiment configs are used on the test set (default hparams are also used for comparison).
