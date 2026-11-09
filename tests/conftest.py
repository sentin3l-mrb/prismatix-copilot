"""Marks the pytest environment so tools.run_tests() can never recurse:
the agent's `run_tests` tool spawns pytest; if that tool is itself invoked
from a test (directly or by a model in a scripted loop) it must bail out."""

import os

os.environ["PRISMATIX_IN_PYTEST"] = "1"
