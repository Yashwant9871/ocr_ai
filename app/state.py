import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RuntimeState:
    current_count: int = 0
    current_frame: Optional[np.ndarray] = None
    lock: threading.Lock = field(default_factory=threading.Lock)
