#!/bin/bash

# Simple linting script using ruff
# Usage: ./lint.sh [check|fix|format]

# Key features:
# Aggressive auto-fixing with unsafe fixes enabled
# Research-friendly configuration (ignores overly strict rules)
# Handles common patterns like l for labels, bare except, etc.
# Whitespace cleanup automatically fixed by formatter
# Import positioning and style issues auto-fixed
# Rules ignored for research code:
# Type annotations, PathLib modernization, bare except
# Mixed case variables (common in data science)
# Module-level imports (for conditional imports)
# Various pedantic style rules

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default to check mode
MODE=${1:-check}

case $MODE in
    "check")
        echo -e "${YELLOW}🔍 Running ruff linter (check mode)...${NC}"
        ruff check .
        echo -e "${YELLOW}🔍 Running ruff formatter (check mode)...${NC}"
        ruff format --check .
        echo -e "${GREEN}✅ Linting check complete!${NC}"
        ;;
    "fix")
        echo -e "${YELLOW}🔧 Running ruff linter with autofix (including unsafe fixes)...${NC}"
        ruff check --fix --unsafe-fixes .
        echo -e "${YELLOW}🔧 Running ruff formatter...${NC}"
        ruff format .
        echo -e "${GREEN}✅ Linting and formatting complete!${NC}"
        ;;
    "format")
        echo -e "${YELLOW}🔧 Running ruff formatter only...${NC}"
        ruff format .
        echo -e "${GREEN}✅ Formatting complete!${NC}"
        ;;
    *)
        echo "Usage: $0 [check|fix|format]"
        echo "  check  - Check for issues without fixing (default)"
        echo "  fix    - Fix issues automatically and format"
        echo "  format - Format code only"
        exit 1
        ;;
esac 