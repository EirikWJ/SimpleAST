import sys
from pathlib import Path

# MAR-AST is not an installable package. Its modules import each other as
# "simulator...", "env_wrappers..." and "utils...", so its root must be on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[2]
MARAST_ROOT = REPO_ROOT / "external" / "MAR-AST"

if not (MARAST_ROOT / "simulator").is_dir():
    raise ImportError(f"MAR-AST not found in {MARAST_ROOT}. Run: git submodule update --init")

if str(MARAST_ROOT) not in sys.path:
    sys.path.insert(0, str(MARAST_ROOT))
