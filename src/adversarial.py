"""
Adversarial perturbation demo for the VR lab.

Ties directly to Section 6.6 of the assignment brief: "AI predictions
should be presented as model outputs rather than unquestionable facts...
students should be encouraged to ask why a prediction was produced,
whether the data is suitable and what limitations may affect the result."

This is a deliberately small, explainable attack -- NOT the full
adversarial ML machinery from the honours research (FGSM/PGD proper
require a differentiable model; RandomForest isn't). Instead this
demonstrates the same underlying idea a first-year can follow:
a small, carefully-chosen nudge to the inputs can shift the model's
prediction disproportionately -- which is exactly the point Section
6.6 wants students to internalize.
"""

from dataclasses import dataclass

import numpy as np

from .model import TrainedModel, predict_with_overrides


@dataclass
class AttackResult:
    original_prediction: float
    attacked_prediction: float
    shift_dollars: float
    shift_percent: float
    perturbed_inputs: dict
    original_inputs: dict


def _clip(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


def run_adversarial_attack(
    trained: TrainedModel,
    baseline_inputs: dict,
    baseline_close: float,
    epsilon: float = 0.15,
    input_bounds: dict | None = None,
) -> AttackResult:
    """
    Finds the single-feature perturbation (within epsilon of each
    feature's own scale) that shifts the prediction the MOST -- a
    simplified stand-in for a gradient-based attack, using finite
    differences instead of a real gradient (RandomForest has none).

    For each feature, nudge it up and down by `epsilon` fraction of
    its allowed range, see which nudge moves the prediction furthest
    from the original, and report the worst one found across all
    features. This is honest about being a search, not a true
    adversarial gradient method -- say so if asked in the report.

    input_bounds: {feature: (min, max)} -- defaults to reasonable
    bounds matching the scene's slider ranges if not supplied.
    """
    if input_bounds is None:
        input_bounds = {
            "volume_norm": (0, 100),
            "volatility": (0, 100),
            "momentum": (-1, 1),
        }

    original_result = predict_with_overrides(trained, baseline_inputs, baseline_close)
    original_prediction = original_result["prediction"]

    worst_shift = 0.0
    worst_inputs = dict(baseline_inputs)

    for feature in trained.feature_columns:
        min_val, max_val = input_bounds[feature]
        feature_range = max_val - min_val
        nudge = feature_range * epsilon

        for direction in (1, -1):
            candidate = dict(baseline_inputs)
            candidate[feature] = _clip(
                baseline_inputs[feature] + direction * nudge, min_val, max_val
            )

            result = predict_with_overrides(trained, candidate, baseline_close)
            shift = abs(result["prediction"] - original_prediction)

            if shift > worst_shift:
                worst_shift = shift
                worst_inputs = candidate

    attacked_result = predict_with_overrides(trained, worst_inputs, baseline_close)
    attacked_prediction = attacked_result["prediction"]
    shift_dollars = attacked_prediction - original_prediction
    shift_percent = (shift_dollars / original_prediction) * 100 if original_prediction else 0.0

    return AttackResult(
        original_prediction=original_prediction,
        attacked_prediction=attacked_prediction,
        shift_dollars=round(shift_dollars, 2),
        shift_percent=round(shift_percent, 2),
        perturbed_inputs=worst_inputs,
        original_inputs=baseline_inputs,
    )
