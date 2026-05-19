"""
_eval_cluster_title gating + fallback contract.

The cheap-LLM title is EE-only and best-effort. The load-bearing
guarantee is that cluster creation NEVER breaks on its account:

  * EE present + good title  -> use it
  * EE present + None        -> deterministic _extract_title
  * EE present + raises      -> deterministic _extract_title
  * EE absent (OSS)          -> deterministic _extract_title

The last is the OSS-safety contract: no ee module, no crash.
"""

import sys
from unittest.mock import patch

from tracer.queries.eval_clustering import _eval_cluster_title, _extract_title

REASONING = (
    "The verdict is Fail because the speech has an unnatural, robotic "
    "rhythm and pacing throughout the call."
)


def test_uses_llm_title_when_available():
    with patch(
        "ee.agenthub.trace_scanner.eval_cluster_title.generate_eval_cluster_title",
        return_value="Robotic, unnatural speech rhythm",
    ):
        assert (
            _eval_cluster_title("prosody_and_intonation", REASONING)
            == "Robotic, unnatural speech rhythm"
        )


def test_falls_back_when_llm_returns_none():
    with patch(
        "ee.agenthub.trace_scanner.eval_cluster_title.generate_eval_cluster_title",
        return_value=None,
    ):
        assert _eval_cluster_title("prosody_and_intonation", REASONING) == _extract_title(
            REASONING
        )


def test_falls_back_when_llm_raises():
    with patch(
        "ee.agenthub.trace_scanner.eval_cluster_title.generate_eval_cluster_title",
        side_effect=RuntimeError("gateway down"),
    ):
        assert _eval_cluster_title("prosody_and_intonation", REASONING) == _extract_title(
            REASONING
        )


def test_oss_safety_no_ee_module_no_crash():
    # Setting the module to None in sys.modules makes `import` raise
    # ImportError — simulates an OSS deployment with no ee package.
    with patch.dict(sys.modules, {"ee.agenthub.trace_scanner.eval_cluster_title": None}):
        assert _eval_cluster_title("prosody_and_intonation", REASONING) == _extract_title(
            REASONING
        )
