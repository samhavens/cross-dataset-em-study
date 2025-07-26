# Potential Improvements for Entity Matching System

## 1. Message Injection for Real-Time Optimization

**Problem**: Currently, Claude Code optimization sessions run for 30+ minutes without ability for human intervention or guidance.

**Solution**: Add CLI-based message injection capability to send guidance to running Claude Code conversations.

**Implementation**:
- Message queue system with file-based communication
- `scripts/inject_message.py` CLI tool for real-time intervention
- Integration with `ClaudeSDKClient` in simplified agentic generator
- Support for different message types: user feedback, interrupt signals, guidance prompts

**Benefits**:
- Real-time course correction during long optimization runs
- Human-in-the-loop optimization for better results
- Ability to stop ineffective exploration paths early

## 2. Persistent Experiment Logs with Version Selection

**Problem**: RunExperiment tool doesn't maintain history of prompt/weight combinations, making it hard to revert to better configurations when scores are tied.

**Solution**: Create persistent experiment log system that tracks all configurations and their results.

**Implementation**:
- Enhanced RunExperiment tool logging all prompt versions and hyperparameters
- Persistent JSON/SQLite database of experiment results
- Version selection mechanism for tied scores based on qualitative factors
- Ability to replay any previous configuration exactly

**Benefits**:
- Better decision making when F1 scores are close
- Full experiment reproducibility
- Historical analysis of what works across datasets
- Rollback capability when optimization goes wrong

## 3. Modular Architecture Refactor

**Problem**: Current codebase has tight coupling between evaluation, retrieval models, and generative models, making it hard to experiment with different components.

**Solution**: Complete modular refactor with clean interfaces and dependency injection.

**Components**:
- **Evaluation Module**: Separate from matching logic, pluggable metrics
- **Retrieval Models**: Abstract interface for candidate generation (trigram, semantic, learned embeddings)
- **Generative Models**: Abstract interface for LLM providers (OpenAI, Anthropic, local models)
- **Matching Pipeline**: Orchestrates components without tight coupling
- **Configuration System**: Unified config management across all components

**Requirements**:
- Extensive test suite for each module
- Clear interfaces and contracts
- Easy swapping of components
- Backward compatibility with existing experiments

**Benefits**:
- Easy experimentation with new models
- Better testing and validation
- Cleaner codebase maintenance
- Easier onboarding for new contributors

## 4. Persistent IPython Kernel for Instant Reweighting

**Problem**: Each hyperparameter adjustment requires full pipeline restart, making optimization slow.

**Solution**: Use persistent IPython kernel to maintain loaded data and models in memory.

**Implementation**:
- IPython MCP server integration for stateful execution
- Pre-load datasets, embeddings, and candidate caches in kernel
- Instant reweighting without data reload
- Incremental evaluation on subset for rapid iteration

**Technical Details**:
- Leverage existing IPython MCP server setup
- Cache candidate generation results
- Smart invalidation when parameters change significantly
- Memory management for large datasets

**Benefits**:
- Sub-second hyperparameter testing
- Interactive optimization experience
- Better exploration of parameter space
- Reduced API costs during development

## Implementation Priority

1. **Message Injection** - High impact, moderate effort, immediate benefit for current optimization sessions
2. **Persistent Logs** - Medium impact, low effort, improves decision making quality
3. **IPython Kernel** - High impact, medium effort, dramatically speeds up development cycle
4. **Modular Refactor** - High impact, high effort, long-term maintainability and extensibility

## Notes

- Message injection system design already exists from previous planning session
- IPython MCP server is already installed and configured
- Modular refactor should be done incrementally to avoid breaking existing functionality
- All improvements should maintain backward compatibility with current experiment results