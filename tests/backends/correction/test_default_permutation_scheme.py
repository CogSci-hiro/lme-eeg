import inspect

from lmeeeg.backends.correction.maxstat_backend import MaxStatCorrectionBackend
from lmeeeg.backends.correction.mne_cluster_backend import MNEClusterCorrectionBackend
from lmeeeg.backends.correction.mne_tfce_backend import MNETFCorrectionBackend


def test_correction_backends_default_to_within_subject_permutation() -> None:
    for backend_cls in (
        MaxStatCorrectionBackend,
        MNEClusterCorrectionBackend,
        MNETFCorrectionBackend,
    ):
        parameter = inspect.signature(backend_cls.run).parameters["permutation_scheme"]
        assert parameter.default == "within_subject"
