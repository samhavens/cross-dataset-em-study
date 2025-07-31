#!/usr/bin/env python3
"""
Hybrid Matcher Prompt Template

This file defines the prompt as a simple dict that MCP tools can easily manipulate.
Each section has a title, description, and either ordered or unordered lists.
"""

import json
import os

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PromptSection:
    """A section of the prompt with title, description, and content lists."""
    title: str
    description: str = ""
    ordered_list: Optional[List[str]] = None
    unordered_list: Optional[List[str]] = None

    def render(self) -> str:
        """Render the section as formatted text."""
        result = f"{self.title}\n"

        if self.description:
            result += f"{self.description}\n\n"

        if self.ordered_list:
            for i, item in enumerate(self.ordered_list, 1):
                result += f"{i}. {item}\n"

        if self.unordered_list:
            for item in self.unordered_list:
                result += f"- {item}\n"

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptSection':
        """Create PromptSection from dictionary."""
        return cls(
            title=data["title"],
            description=data.get("description", ""),
            ordered_list=data.get("ordered_list"),
            unordered_list=data.get("unordered_list")
        )


# Prompt data as a simple dict that MCP tools can manipulate
PROMPT_DATA = {
    "sections": [
        {
            "title": "ENTITY MATCHING TASK",
            "description": "You are an expert at entity matching. Your task is to find the candidate that refers to the same real-world entity as the left record.",
            "unordered_list": [
                "Two records match if they refer to the same entity, even if:",
                "They have different formatting or spellings",
                "One has more/less information than the other",
                "They use different abbreviations or representations"
            ]
        },
        {
            "title": "COMPARISON CRITERIA",
            "description": "Compare the left record against the candidate. Look for:",
            "ordered_list": [
                "Same entity name (allowing for variations in spelling/format)",
                "Matching key identifiers (IDs, codes, etc.)",
                "Consistent attribute values where they overlap",
                "No contradictory information"
            ]
        },
        {
            "title": "REASONING PROCESS",
            "description": "Think step by step and identify if the candidate represents the same entity."
        },
        {
            "title": "RESPONSE FORMAT REQUIREMENTS",
            "description": "Your response must follow this exact format:",
            "unordered_list": [
                "You MUST respond with ONLY a single integer number",
                "If the candidate matches the left record: respond with the candidate number",
                "If the candidate does NOT match the left record: respond with \"-1\"",
                "DO NOT include any words, explanations, or other text",
                "DO NOT include quotes, brackets, or punctuation",
                "Examples of CORRECT responses: `7` or `-1`",
                "Examples of INCORRECT responses: \"no match\", \"the movie babel\", \"candidate 0\""
            ]
        }
    ]
}


def get_prompt_data() -> Dict[str, Any]:
    """Get the current prompt data structure."""
    return PROMPT_DATA


def update_prompt_data(new_data: Dict[str, Any]) -> None:
    """Update the prompt data structure and persist to file."""
    global PROMPT_DATA
    PROMPT_DATA = new_data

    # Persist to file for cross-process use
    prompt_file = "results/temp/prompt_data.json"
    os.makedirs(os.path.dirname(prompt_file), exist_ok=True)
    with open(prompt_file, 'w') as f:
        json.dump(new_data, f, indent=2)


def build_prompt(
    left_record: dict,
    candidates_text: str,
    best_idx: int,
    prompt_data: Optional[Dict[str, Any]] = None,
    additional_guidance: Optional[List[str]] = None
) -> str:
    """
    Build the complete prompt from the data structure.

    Args:
        left_record: The record to match
        candidates_text: Formatted candidate text
        best_idx: Index of the best candidate
        prompt_data: Optional prompt data structure (if None, uses default + MCP file)
        additional_guidance: Optional list of additional guidance strings

    Returns:
        Complete formatted prompt
    """
    # Use provided prompt_data or fall back to current behavior for backward compatibility
    if prompt_data is None:
        # Legacy behavior: Load prompt data from file if it exists (for cross-process persistence)
        prompt_data = PROMPT_DATA
        prompt_file = "results/temp/prompt_data.json"

        if os.path.exists(prompt_file):
            try:
                with open(prompt_file) as f:
                    prompt_data = json.load(f)
            except Exception:
                # Fall back to default if file is corrupted
                prompt_data = PROMPT_DATA

    sections = []

    # Convert dict sections to PromptSection objects and add dynamic content
    for section_data in prompt_data["sections"]:
        if section_data["title"] == "ENTITY MATCHING TASK":
            sections.append(PromptSection.from_dict(section_data))

            # Insert records section after base instructions
            records_section = PromptSection(
                title="RECORDS TO COMPARE",
                description=f"""LEFT RECORD:
{json.dumps(left_record, ensure_ascii=False)}

CANDIDATES:
{candidates_text}"""
            )
            sections.append(records_section)

        elif section_data["title"] == "REASONING PROCESS":
            # Insert additional guidance before reasoning if provided
            if additional_guidance:
                guidance_section = PromptSection(
                    title="ADDITIONAL GUIDANCE",
                    description="Apply these specific rules:",
                    unordered_list=additional_guidance
                )
                sections.append(guidance_section)

            sections.append(PromptSection.from_dict(section_data))

        elif section_data["title"] == "RESPONSE FORMAT REQUIREMENTS":
            # Update response format with actual candidate index
            updated_data = section_data.copy()
            if updated_data.get("unordered_list"):
                updated_data["unordered_list"] = [
                    item.replace("candidate number", str(best_idx))
                         .replace("Examples of CORRECT responses: 0", f"Examples of CORRECT responses: {best_idx}")
                    for item in updated_data["unordered_list"]
                ]
            sections.append(PromptSection.from_dict(updated_data))

        else:
            sections.append(PromptSection.from_dict(section_data))

    # Render all sections
    prompt_parts = []
    for section in sections:
        prompt_parts.append(section.render())

    # Add final answer prompt
    prompt_parts.append("YOUR ANSWER (integer only):")

    return "\n".join(prompt_parts)
