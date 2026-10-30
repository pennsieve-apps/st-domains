"""
Container entry point: fixes the thread count from THREADS (or the cgroup CPU
quota) before numpy is imported, then runs main() once.
"""
import sys

import processor.threads  # noqa: F401  (sets the BLAS / OpenMP / numba thread variables)
from processor.main import main  # noqa: E402

sys.exit(main())
