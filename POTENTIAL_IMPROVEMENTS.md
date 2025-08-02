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
- **Refactor RunExperiment**: Accept weights and prompt_data as parameters instead of using WriteWeights/WritePrompt
- Enhanced RunExperiment tool logging all prompt versions and hyperparameters with auto-incrementing experiment IDs
- Persistent JSON/SQLite database of experiment results
- FetchBest functionality to return optimal experiment configuration
- Version selection mechanism for tied scores based on qualitative factors
- Ability to replay any previous configuration exactly

**API Design**:
```
RunExperiment(weights={semantic: 0.3, trigram: 0.5, syntactic: 0.2}, prompt_data={...}, experiment_id=1)
RunExperiment(weights={semantic: 0.6, trigram: 0.2, syntactic: 0.2}, prompt_data={...}, experiment_id=2)  
FetchBest() → returns best experiment configuration and results
```

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

## 5. Fix Rules File Race Condition

**Problem**: All datasets share the same `results/temp/generated_rules.json` file, causing race conditions when running multiple datasets in parallel.

**Impact**: 
- iTunes Amazon gets Beer's configuration
- Optimization results are inconsistent and unpredictable
- "Beautiful" optimization runs randomly break due to config corruption

**Root Cause**: Hardcoded shared file path in 24+ locations across codebase

**Solution**: Make rules files dataset-specific throughout entire pipeline

**Implementation**:
```python
# Current (race condition):
rules_file = "results/temp/generated_rules.json"

# Fixed (dataset-specific):
rules_file = f"results/temp/generated_rules_{dataset}.json"
```

**Files Requiring Changes**:
- `src/mcp_servers/entity_matching_server.py` - All MCP tools need dataset parameter
- `src/experiments/simplified_agentic_generator.py` - Pass dataset to MCP tools
- `run_complete_pipeline.py` - Update rules file paths
- 20+ script files with hardcoded paths

**Technical Requirements**:
- Thread dataset parameter through all MCP tool functions
- Update MCP tool signatures: `write_weights_tool(dataset, ...)`
- Modify simplified agentic generator to pass dataset context
- Update all hardcoded file references

**Benefits**:
- Eliminates rules file race conditions
- Enables truly parallel dataset processing
- Consistent optimization results
- No more random "beautiful run" failures

## 6. Modular Embedding Architecture

**Problem**: Current embedding implementation is tightly coupled to SentenceTransformers with `all-MiniLM-L6-v2`, limiting experimentation with different embedding approaches.

**Current State**: 
- Single hardcoded model in `hybrid_matcher.py`
- Direct SentenceTransformer API usage throughout codebase
- Basic cosine similarity with linear search
- No abstraction layer for different embedding providers

**Proposed Architecture**:
```python
class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> np.ndarray: pass
    
    @abstractmethod  
    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float: pass

class LocalSentenceTransformerProvider(EmbeddingProvider):
    # Current implementation with all-MiniLM-L6-v2
    
class DatabricksVectorSearchProvider(EmbeddingProvider):
    # Databricks-hosted embedding + vector search
```

**Local vs Databricks Comparison**:

| Aspect | Local (MacBook Pro) | Databricks VectorSearch |
|--------|-------------------|------------------------|
| **Embedding Speed** | ~200 records/sec (M1/M2) | ~1000+ records/sec (GPU clusters) |
| **Memory Usage** | 500MB model + embeddings | Offloaded to cluster |
| **Similarity Search** | Linear O(n) cosine | Approximate nearest neighbors O(log n) |
| **Cold Start** | 2-3 seconds model load | Network latency + cluster warmup |
| **Cost per Query** | Free after hardware | $0.0001-0.001 per query |
| **Offline Capability** | Full offline | Requires network |
| **Model Flexibility** | Any SentenceTransformer | Databricks-supported models only |
| **Batch Processing** | Limited by RAM | Scales with cluster size |

**Implementation Plan**:
1. Extract embedding logic into abstract `EmbeddingProvider` interface
2. Refactor existing code to use provider pattern
3. Implement `LocalSentenceTransformerProvider` (current behavior)
4. Add `DatabricksVectorSearchProvider` with vector index integration
5. Configuration-driven provider selection

**Benefits**:
- Easy A/B testing between embedding approaches
- Scales beyond single-machine memory limits
- Future-proof for new embedding models
- Production deployment flexibility

## Implementation Priority

1. **Rules File Race Condition** - **CRITICAL**, high impact, moderate effort, prevents optimization corruption
2. **Message Injection** - High impact, moderate effort, immediate benefit for current optimization sessions
3. **Persistent Logs** - Medium impact, low effort, improves decision making quality
4. **IPython Kernel** - High impact, medium effort, dramatically speeds up development cycle
5. **Modular Embedding Architecture** - Medium impact, medium effort, enables production scaling
6. **Modular Refactor** - High impact, high effort, long-term maintainability and extensibility

## Notes

- Message injection system design already exists from previous planning session
- IPython MCP server is already installed and configured
- Modular refactor should be done incrementally to avoid breaking existing functionality
- All improvements should maintain backward compatibility with current experiment results