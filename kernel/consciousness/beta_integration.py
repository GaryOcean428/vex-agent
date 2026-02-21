"""
β-Attention Integration — Hook into ConsciousnessLoop
=====================================================

Provides a factory function and the minimal integration points
needed to wire BetaAttentionTracker into the existing consciousness
loop without rewriting the 800-line loop.py.

INTEGRATION GUIDE (6 insertions into loop.py):
---------------------------------------------

1. IMPORT (top of loop.py, after emotions imports):
   from .beta_integration import create_beta_tracker

2. INIT (in ConsciousnessLoop.__init__, after self.learner = ...):
   self.beta_tracker = create_beta_tracker(settings.data_dir)

3. RECORD (in ConsciousnessLoop._process, after self.learner.record(...)):
   self.beta_tracker.record(
       context_length=len(task.content),
       kappa_eff=self.metrics.kappa,
       phi_before=phi_before,
       phi_after=self.metrics.phi,
       perceive_distance=perceive_distance,
       integration_distance=integration_distance,
       express_distance=express_distance,
       total_distance=total_distance,
       processing_path=processing_path.value,
   )

4. PERSIST (in ConsciousnessLoop._persist_state, add to state dict):
   state["beta_tracker"] = self.beta_tracker.serialize()

5. RESTORE (in ConsciousnessLoop._restore_state, after kernel restore):
   beta_data = data.get("beta_tracker")
   if beta_data:
       self.beta_tracker.restore(beta_data)

6. METRICS (in ConsciousnessLoop.get_full_state, add to return dict):
   "beta_tracker": self.beta_tracker.get_summary(),
"""

from pathlib import Path

from .beta_tracker import BetaAttentionTracker


def create_beta_tracker(data_dir: str) -> BetaAttentionTracker:
    """Create a BetaAttentionTracker with persistence path."""
    persist_path = Path(data_dir) / "beta_attention_state.json"
    persist_path.parent.mkdir(parents=True, exist_ok=True)
    return BetaAttentionTracker(persist_path=persist_path)
