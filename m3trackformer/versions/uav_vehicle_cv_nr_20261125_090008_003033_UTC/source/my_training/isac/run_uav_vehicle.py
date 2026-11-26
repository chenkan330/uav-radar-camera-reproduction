"""Run the paired UAV/vehicle experiment in the project's Python environment."""
import os
from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    training = Path(__file__).resolve().parents[1]
    interpreter = training.parent / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if interpreter.is_file() and interpreter.resolve() != Path(sys.executable).resolve():
        raise SystemExit(subprocess.call([str(interpreter), "-X", "utf8", __file__, *sys.argv[1:]]))
    os.environ.setdefault("MPLCONFIGDIR", str(training.parent / ".cache" / "matplotlib"))
    sys.path.insert(0, str(training))
    from core.uav_vehicle_experiment import main

    main()
