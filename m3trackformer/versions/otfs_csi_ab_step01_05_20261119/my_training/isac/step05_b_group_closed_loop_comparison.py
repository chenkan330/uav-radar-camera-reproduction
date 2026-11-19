"""运行 B 组第 5 步，与冻结 V6 做三目标闭环配对比较。"""
import os
from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    training = Path(__file__).resolve().parents[1]
    python = training.parent/".venv"/"Scripts"/"python.exe"
    if python.is_file() and Path(sys.executable).resolve() != python.resolve():
        raise SystemExit(subprocess.call([str(python), "-X", "utf8", __file__, *sys.argv[1:]]))
    os.environ.setdefault("MPLCONFIGDIR", str(training.parent/".cache"/"matplotlib"))
    sys.path.insert(0, str(training))
    from core.isac_b_closed_loop_comparison import main

    main()
