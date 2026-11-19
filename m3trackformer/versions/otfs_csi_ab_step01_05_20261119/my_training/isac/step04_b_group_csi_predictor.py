"""运行 B 组第 4 步，采集分区数据并训练未来 CSI 预测器。"""
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
    from core.isac_b_csi_predictor import main

    main()
