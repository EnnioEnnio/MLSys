"""P1: registering an adapter before the first build_backbone() must not suppress
the built-in adapters.

Run in a fresh interpreter so the built-ins are genuinely un-imported at the
point register_adapter() is first called — the exact misuse sequence that the
old `if _ADAPTERS: return` guard broke. In-process this can't be reproduced
once any earlier test has imported the adapters.
"""

from __future__ import annotations

import subprocess
import sys

_SCRIPT = """
from mlsys.models import registry

# Misuse: a peer registers their adapter before any build_backbone() call.
registry.register_adapter("dummy_p1", lambda spec, device: object())
registry._ensure_adapters_registered()

loaders = set(registry._ADAPTERS)
builtins = {"sentence_transformers", "transformers_encoder", "model2vec"}
assert builtins <= loaders, f"built-ins missing after early register_adapter: {loaders}"
print("OK")
"""


def test_builtins_survive_early_register_adapter() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
