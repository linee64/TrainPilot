"""TrainPilot public API.

Step timing, diagnosis, the mixed-precision comparison, and the one-page
report are available. ``Pilot`` is not implemented yet.
"""

from trainpilot.diagnose import Diagnosis, diagnose
from trainpilot.experiments import ExperimentResult, compare_mixed_precision
from trainpilot.measure import StepMeasurement, measure_batches, measure_steps
from trainpilot.report import render_report

__version__ = "0.1.0"

__all__ = [
    "Diagnosis",
    "ExperimentResult",
    "StepMeasurement",
    "compare_mixed_precision",
    "diagnose",
    "measure_batches",
    "measure_steps",
    "render_report",
    "__version__",
]
