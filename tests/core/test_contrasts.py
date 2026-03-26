import numpy as np

from lmeeeg.core.contrasts import effect_to_contrast


def test_effect_to_contrast() -> None:
    contrast = effect_to_contrast("condition[T.B]", ["Intercept", "condition[T.B]", "latency"])
    assert np.array_equal(contrast, np.array([0.0, 1.0, 0.0]))
