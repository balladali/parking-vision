from pathlib import Path
import subprocess
import sys

import pytest

from parking_vision.cli import main


def test_console_entrypoint() -> None:
    executable = Path(sys.executable).with_name("parking-vision.exe" if sys.platform == "win32" else "parking-vision")
    result = subprocess.run([str(executable), "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "run" in result.stdout
    assert "validate" in result.stdout


def test_help_does_not_import_yolo() -> None:
    code = "import sys; from parking_vision.cli import main\ntry: main(['--help'])\nexcept SystemExit: pass\nassert 'ultralytics' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def test_validate(capsys: pytest.CaptureFixture) -> None:
    path = Path(__file__).parents[1] / "configs/example.yaml"
    assert main(["validate", "--config", str(path)]) == 0
    captured = capsys.readouterr()
    assert not captured.out
    assert "Valid configuration" in captured.err


def test_config_error(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("spots: [", encoding="utf-8")
    assert main(["validate", "--config", str(path)]) == 1
    assert "ERROR" in capsys.readouterr().err


def test_reject_zero_frames() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["run", "--config", "unused.yaml", "--max-frames", "0"])
    assert exc.value.code == 2
