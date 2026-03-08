"""FastAPI application for the Varaha OpenEnv environment."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openenv.core.env_server import create_app

from openenv_wrapper.models import VarahaAction, VarahaObservation
from openenv_wrapper.varaha_environment import VarahaEnvironment

app = create_app(
    VarahaEnvironment,
    VarahaAction,
    VarahaObservation,
    env_name="varaha",
)
