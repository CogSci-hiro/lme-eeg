import pandas as pd
import pytest

from lmeeeg.core.coding import validate_variable_types


def test_validate_variable_types_rejects_unknown_columns() -> None:
    metadata = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError):
        validate_variable_types(metadata=metadata, variable_types={"missing": "numeric"})
