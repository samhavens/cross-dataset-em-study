# Entity Matching MCP Server

This MCP server provides specialized tools for entity matching rule generation that enforce correct formats and prevent common errors.

## Features

The server provides 6 custom tools that replace the generic Read/Write/Bash tools:

### 🔧 **WriteRules**
- **Purpose**: Save entity matching rules in the enforced correct format
- **Enforces**: 
  - Exact filename: `results/temp/generated_rules.json`
  - Correct JSON keys: `candidate_rules`, `score_rules`, `decision_rules`, `weight_rules`, `pipeline_rules`
  - 3-weight system: Requires `semantic_weight`, `trigram_weight`, `syntactic_weight`
  - Weight validation: Must sum to ~1.0
- **Prevents**: Wrong filenames, wrong JSON format, missing weights

### 🧪 **TestRules** 
- **Purpose**: Test rules and return structured performance results
- **Features**:
  - Runs pipeline with generated rules
  - Returns F1, precision, recall metrics
  - Uses validation data by default
  - Configurable test limits
- **Prevents**: Manual command construction, test data leakage

### 📊 **ReadSampleData**
- **Purpose**: Read and format sample data for analysis
- **Features**:
  - Structured data summary
  - Error pattern examples
  - Performance metrics
- **Prevents**: Manual JSON parsing, format confusion

### 📋 **ReadInstructions**
- **Purpose**: Provide clear, structured task instructions
- **Features**:
  - Dataset-specific targets
  - Clear workflow steps  
  - Critical requirements
- **Prevents**: Prompt overload, missing requirements

### 📈 **AnalyzePerformance**
- **Purpose**: Analyze current vs target performance and suggest improvements
- **Features**:
  - Gap analysis
  - Tailored recommendations
  - Next step guidance
- **Prevents**: Unclear optimization direction

### 📊 **GetBaseline**
- **Purpose**: Get current baseline performance metrics
- **Features**:
  - Loads existing dev results
  - Shows current F1/precision/recall
  - Multiple file source fallbacks
- **Prevents**: Manual metric calculation

## Installation & Setup

### 1. Install Dependencies
```bash
cd src/mcp_servers
pip install mcp
```

### 2. Add to Claude Desktop Configuration

Add this to your Claude Desktop configuration file (usually `~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "entity-matching": {
      "command": "python",
      "args": [
        "src/mcp_servers/entity_matching_server.py"
      ],
      "env": {
        "PYTHONPATH": "."
      }
    }
  }
}
```

**Important**: This configuration assumes you're running from the project root directory.

### 3. Restart Claude Desktop

After adding the configuration, restart Claude Desktop to load the MCP server.

## Usage

Once connected, Claude will have access to these tools instead of generic ones:

```
✅ Available Tools:
- WriteRules: Save rules in correct format with 3 weights
- TestRules: Test performance on validation data  
- ReadSampleData: Examine current errors and patterns
- ReadInstructions: Get clear task guidance
- AnalyzePerformance: Get improvement suggestions
- GetBaseline: Check current metrics
```

## Example Workflow

```
1. ReadInstructions(dataset="beer") → Get task overview
2. ReadSampleData(dataset="beer") → Understand error patterns  
3. GetBaseline(dataset="beer") → Check current performance
4. WriteRules(semantic_weight=0.6, trigram_weight=0.2, syntactic_weight=0.2, 
              candidate_rules=[...], score_rules=[...]) → Save rules
5. TestRules(dataset="beer") → Check performance
6. AnalyzePerformance(current_f1=0.85, target_f1=95.3, dataset="beer") → Get suggestions
7. Repeat steps 4-6 until target achieved
```

## Benefits

- **No Format Errors**: Tools enforce correct JSON structure and filename
- **No Missing Weights**: WriteRules requires all 3 weights
- **No Test Leakage**: TestRules uses validation by default  
- **Clear Workflow**: Tools guide the optimization process
- **Structured Results**: All outputs are properly formatted
- **Error Prevention**: Validation prevents common mistakes

## Troubleshooting

### MCP Server Not Loading
- Check Claude Desktop configuration path is absolute
- Verify Python path includes the project directory
- Restart Claude Desktop after configuration changes

### Tool Errors
- WriteRules: Check weights sum to ~1.0
- TestRules: Ensure rules file exists and dataset is valid
- All tools: Check you're in the correct working directory

## Development

To modify the MCP server:
1. Edit `entity_matching_server.py`
2. Restart Claude Desktop to reload
3. Test tools in Claude Desktop chat

The server uses the official MCP Python SDK and follows MCP protocol standards.