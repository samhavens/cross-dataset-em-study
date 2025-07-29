#!/usr/bin/env python
"""
Experiment Registry for Stage-Level Experiment Tracking

This module provides a registry system for tracking all experiments within a pipeline run,
enabling proper linkage between development, Claude optimization, baseline, and enhanced stages.
"""

import json
import uuid

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .experiment_config import ExperimentConfig


@dataclass
class ExperimentEntry:
    """Individual experiment entry in the registry"""

    experiment_config: ExperimentConfig
    stage: str  # "dev", "claude_optimization", "3A_baseline", "3B_enhanced"
    results: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "experiment_id": self.experiment_config.experiment_id,
            "stage": self.stage,
            "config": self.experiment_config.to_dict(),
            "results": self.results,
            "timestamp": self.timestamp,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentEntry":
        """Create from dictionary"""
        config = ExperimentConfig.from_dict(data["config"])
        return cls(
            experiment_config=config,
            stage=data["stage"],
            results=data.get("results"),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            notes=data.get("notes"),
        )


class ExperimentRegistry:
    """
    Registry for tracking all experiments within a pipeline run.

    Maintains complete experiment genealogy and enables proper linkage
    between stages (dev → claude experiments → 3A → 3B).
    """

    def __init__(self, pipeline_run_id: str = None):
        """Initialize registry for a pipeline run"""
        self.pipeline_run_id = pipeline_run_id or str(uuid.uuid4())[:8]
        self.created_at = datetime.now().isoformat()
        self.experiments: List[ExperimentEntry] = []
        self._registry_dir = Path("results/registries")
        self._registry_dir.mkdir(parents=True, exist_ok=True)

    def register_experiment(
        self, experiment_config: ExperimentConfig, stage: str, results: Dict[str, Any] = None, notes: str = None
    ) -> str:
        """
        Register a new experiment in this pipeline run.

        Args:
            experiment_config: Complete experiment configuration
            stage: Stage identifier (dev, claude_optimization, 3A_baseline, 3B_enhanced)
            results: Experiment results (F1 score, metrics, etc.)
            notes: Optional notes about this experiment

        Returns:
            experiment_id: The ID of the registered experiment
        """
        entry = ExperimentEntry(experiment_config=experiment_config, stage=stage, results=results, notes=notes)

        self.experiments.append(entry)

        # Auto-save registry after each registration
        self.save_registry()

        print(f"📋 Registered experiment {experiment_config.experiment_id} for stage '{stage}'")
        return experiment_config.experiment_id

    def get_experiments_by_stage(self, stage: str) -> List[ExperimentEntry]:
        """Get all experiments for a specific stage"""
        return [exp for exp in self.experiments if exp.stage == stage]

    def get_claude_experiments(self) -> List[ExperimentEntry]:
        """Get all Claude optimization experiments"""
        return self.get_experiments_by_stage("claude_optimization")

    def get_best_claude_experiment(self) -> Optional[ExperimentConfig]:
        """
        Get Claude's best experiment based on F1 score.

        Returns:
            Best performing Claude experiment config, or None if no Claude experiments
        """
        claude_experiments = self.get_claude_experiments()

        if not claude_experiments:
            return None

        # Filter experiments with results containing F1 scores
        scored_experiments = []
        for exp in claude_experiments:
            if exp.results and isinstance(exp.results, dict):
                f1_score = None
                # Try different possible F1 score keys
                for key in ["f1", "enhanced_f1", "f1_score", "score"]:
                    if key in exp.results:
                        f1_score = exp.results[key]
                        break

                if f1_score is not None:
                    scored_experiments.append((exp, f1_score))

        if not scored_experiments:
            # If no scored experiments, return the most recent one
            return claude_experiments[-1].experiment_config

        # Return config of experiment with highest F1 score
        best_exp, best_score = max(scored_experiments, key=lambda x: x[1])
        print(f"🏆 Selected best Claude experiment {best_exp.experiment_config.experiment_id} with F1={best_score:.3f}")
        return best_exp.experiment_config

    def get_dev_experiment(self) -> Optional[ExperimentConfig]:
        """Get the development experiment configuration"""
        dev_experiments = self.get_experiments_by_stage("dev")
        return dev_experiments[0].experiment_config if dev_experiments else None

    def get_experiment_genealogy(self) -> Dict[str, Any]:
        """
        Generate experiment genealogy showing the flow:
        dev → claude experiments → 3A → 3B
        """
        genealogy = {"pipeline_run_id": self.pipeline_run_id, "created_at": self.created_at, "stages": {}}

        # Group experiments by stage
        for stage in ["dev", "claude_optimization", "3A_baseline", "3B_enhanced"]:
            stage_experiments = self.get_experiments_by_stage(stage)
            genealogy["stages"][stage] = [
                {
                    "experiment_id": exp.experiment_config.experiment_id,
                    "timestamp": exp.timestamp,
                    "f1_score": self._extract_f1_score(exp.results) if exp.results else None,
                    "notes": exp.notes,
                }
                for exp in stage_experiments
            ]

        # Add best Claude experiment indicator
        best_claude = self.get_best_claude_experiment()
        if best_claude:
            genealogy["best_claude_experiment_id"] = best_claude.experiment_id

        return genealogy

    def _extract_f1_score(self, results: Dict[str, Any]) -> Optional[float]:
        """Extract F1 score from results dict"""
        if not results:
            return None

        for key in ["f1", "enhanced_f1", "f1_score", "score"]:
            if key in results:
                return results[key]
        return None

    def save_registry(self) -> Path:
        """Save complete registry to file"""
        registry_data = {
            "pipeline_run_id": self.pipeline_run_id,
            "created_at": self.created_at,
            "experiments": [exp.to_dict() for exp in self.experiments],
        }

        registry_path = self._registry_dir / f"pipeline_{self.pipeline_run_id}.json"
        with open(registry_path, "w") as f:
            json.dump(registry_data, f, indent=2)

        return registry_path

    @classmethod
    def load_registry(cls, pipeline_run_id: str) -> "ExperimentRegistry":
        """Load existing registry from file"""
        registry_dir = Path("results/registries")
        registry_path = registry_dir / f"pipeline_{pipeline_run_id}.json"

        if not registry_path.exists():
            raise FileNotFoundError(f"Registry not found: {registry_path}")

        with open(registry_path) as f:
            data = json.load(f)

        registry = cls(pipeline_run_id=data["pipeline_run_id"])
        registry.created_at = data["created_at"]
        registry.experiments = [ExperimentEntry.from_dict(exp_data) for exp_data in data["experiments"]]

        return registry

    @classmethod
    def find_active_pipeline_registry(cls) -> Optional["ExperimentRegistry"]:
        """
        Find the most recently created registry, assuming it's the active pipeline.

        Returns:
            Most recent registry, or None if no registries exist
        """
        registry_dir = Path("results/registries")
        if not registry_dir.exists():
            return None

        registry_files = list(registry_dir.glob("pipeline_*.json"))
        if not registry_files:
            return None

        # Get most recent registry file
        most_recent = max(registry_files, key=lambda p: p.stat().st_mtime)
        pipeline_id = most_recent.stem.replace("pipeline_", "")

        return cls.load_registry(pipeline_id)

    def print_summary(self) -> None:
        """Print human-readable registry summary"""
        print("\n🔬 Experiment Registry Summary")
        print(f"Pipeline Run: {self.pipeline_run_id}")
        print(f"Created: {self.created_at[:19]}")
        print(f"Total Experiments: {len(self.experiments)}")

        # Print by stage
        for stage in ["dev", "claude_optimization", "3A_baseline", "3B_enhanced"]:
            stage_experiments = self.get_experiments_by_stage(stage)
            if stage_experiments:
                print(f"\n📊 {stage.upper()}:")
                for exp in stage_experiments:
                    f1_score = self._extract_f1_score(exp.results)
                    f1_str = f"F1={f1_score:.3f}" if f1_score else "No score"
                    print(f"  • {exp.experiment_config.experiment_id}: {f1_str}")

        # Highlight best Claude experiment
        best_claude = self.get_best_claude_experiment()
        if best_claude:
            print(f"\n🏆 Best Claude Experiment: {best_claude.experiment_id}")

    def __str__(self) -> str:
        """String representation"""
        return f"ExperimentRegistry(pipeline={self.pipeline_run_id}, experiments={len(self.experiments)})"
