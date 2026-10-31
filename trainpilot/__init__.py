"""TrainPilot public API.

Step timing is available as ``measure_steps``. ``Pilot`` is not implemented yet.
"""

from trainpilot.measure import StepMeasurement, measure_steps

__version__ = "0.0.1"

__all__ = ["StepMeasurement", "measure_steps", "__version__"]
