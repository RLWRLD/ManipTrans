from .base import ManipData

import os
from .factory import ManipDataFactory

# This replaces auto_register_data
import main.dataset.ho_tracker

current_dir = os.path.dirname(os.path.abspath(__file__))
base_package = __name__

# ManipDataFactory.auto_register_data(current_dir, base_package)
