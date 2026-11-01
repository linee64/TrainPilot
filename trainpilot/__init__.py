"""TrainPilot public API.

Step timing, the waiting-versus-busy diagnosis, and the mixed-precision
comparison are available. ``Pilot`` is not implemented yet.
"""

from trainpilot.diagnose import Diagnosis, diagnose
from trainpilot.experiments import ExperimentResult, compare_mixed_precision
from trainpilot.measure import StepMeasurement, measure_batches, measure_steps

__version__ = "0.0.1"

__all__ = [
    "Diagnosis",
    "ExperimentResult",
    "StepMeasurement",
    "compare_mixed_precision",
    "diagnose",
    "measure_batches",
    "measure_steps",
    "__version__",
]
