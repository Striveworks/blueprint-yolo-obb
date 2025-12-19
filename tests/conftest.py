import os
from pathlib import Path

from pytest import FixtureRequest, fixture


@fixture(scope="module")
def mutable_datadir(request: FixtureRequest) -> Path:
    # NOTE(s.maddox): using `mutable_datadir` to distinguish from the
    # immutable copy provided by the the `pytest-datadir` plugin.
    moduledir = Path(request.module.__file__).resolve()
    packagedir = moduledir.parent
    mutable_datadir = packagedir / "data"
    return mutable_datadir


@fixture
def enable_self_correction() -> bool:
    return os.environ.get("ENABLE_SELF_CORRECTION") == "true"
