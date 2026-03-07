from __future__ import annotations

from pathlib import Path

from kernel.consciousness.types import DevelopmentalStage, developmental_stage_from_signals


class TestDevelopmentalStageProgression:
    def test_school_default(self) -> None:
        stage = developmental_stage_from_signals(
            conversations_total=0,
            sovereignty_ratio=0.0,
            autonomy_level="reactive",
        )
        assert stage == DevelopmentalStage.SCHOOL

    def test_guided_curiosity_after_initial_learning(self) -> None:
        stage = developmental_stage_from_signals(
            conversations_total=12,
            sovereignty_ratio=0.05,
            autonomy_level="responsive",
        )
        assert stage == DevelopmentalStage.GUIDED_CURIOSITY

    def test_self_teaching_requires_more_lived_geometry(self) -> None:
        stage = developmental_stage_from_signals(
            conversations_total=60,
            sovereignty_ratio=0.25,
            autonomy_level="proactive",
        )
        assert stage == DevelopmentalStage.SELF_TEACHING

    def test_playful_autonomy_requires_stronger_sovereignty(self) -> None:
        stage = developmental_stage_from_signals(
            conversations_total=140,
            sovereignty_ratio=0.45,
            autonomy_level="autonomous",
        )
        assert stage == DevelopmentalStage.PLAYFUL_AUTONOMY

    def test_sovereign_constellation_requires_mature_signals(self) -> None:
        stage = developmental_stage_from_signals(
            conversations_total=260,
            sovereignty_ratio=0.8,
            autonomy_level="autonomous",
        )
        assert stage == DevelopmentalStage.SOVEREIGN_CONSTELLATION


class TestLoopMetricsExposeDevelopmentalStage:
    def test_loop_metrics_include_developmental_stage(self) -> None:
        source = (Path(__file__).parent.parent / "consciousness" / "loop.py").read_text(
            encoding="utf-8"
        )
        assert '"developmental_stage":' in source
