import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))  # lets tests `import fakes`


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Every test gets its own SHELLMATE_HOME, so nothing touches ~/.shellmate."""
    home = tmp_path / "shellmate-home"
    monkeypatch.setenv("SHELLMATE_HOME", str(home))
    return home
