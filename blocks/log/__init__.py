from .log import TrainingLog
from .sqlite import SQLiteLog
from .json import JSONLinesLog

BACKENDS = {
    'python': TrainingLog,
    'sqlite': SQLiteLog,
    'mimir': JSONLinesLog
}
