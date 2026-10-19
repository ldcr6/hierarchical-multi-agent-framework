"""Download HumanEval+ and MBPP+ (EvalPlus releases) into data/evalplus/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hmas.bench.datasets import DATASETS, download  # noqa: E402

for ds in DATASETS:
    print(ds, "->", download(ds))
