import pytest
from pydantic import ValidationError

from tpu_cake.contracts import ProfileExpectation


def test_profile_expectation_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ProfileExpectation.model_validate(
            {"name": "decode", "stage": "steady_decode", "unexpected": True}
        )
