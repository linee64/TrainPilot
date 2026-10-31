"""TrainPilot public API.

Step timing and the waiting-versus-busy diagnosis are available.
``Pilot`` is not implemented yet.
"""

from trainpilot.diagnose import Diagnosis, diagnose
from trainpilot.measure import StepMeasurement, measure_batches, measure_steps

__version__ = "0.0.1"

__all__ = [
    "Diagnosis",
    "StepMeasurement",
    "diagnose",
    "measure_batches",
    "measure_steps",
    "__version__",
]
