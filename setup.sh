#!/bin/bash
# Setup script for cross-dataset entity matching study environment

set -e  # Exit on any error

echo "🚀 Setting up cross-dataset entity matching environment..."

# Check if uv is installed
if command -v uv &> /dev/null; then
    echo "📦 Using uv for fast dependency management..."
    
    # Create virtual environment with uv
    uv venv
    source .venv/bin/activate
    
    # Install dependencies from requirements.txt
    uv pip install -r requirements.txt
    # Install Claude Code SDK and CLI
    uv pip install claude-code-sdk
    if ! command -v claude &> /dev/null; then
        npm install -g @anthropic-ai/claude-code
    fi
    
    echo "✅ Environment setup complete with uv!"
    
elif command -v pip &> /dev/null; then
    echo "📦 Using pip for dependency management..."
    
    # Create virtual environment
    python -m venv .venv
    source .venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install from requirements.txt
    pip install -r requirements.txt
    # Install Claude Code SDK and CLI
    pip install claude-code-sdk
    if ! command -v claude &> /dev/null; then
        npm install -g @anthropic-ai/claude-code
    fi
    
    echo "✅ Environment setup complete with pip!"
    
else
    echo "❌ Neither uv nor pip found. Please install Python and pip first."
    exit 1
fi

echo ""
echo "💡 To activate the environment:"
echo "   source .venv/bin/activate"
echo "   export TOKENIZERS_PARALLELISM=false  # Disable tokenizer warnings"
echo ""
echo "🎯 Then run: ./bin/quick_llm_bench.sh"