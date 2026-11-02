import logging
import json
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
import flowtorch.bijectors as bij

class Config_SD:
    def __init__(self):
        self.alpha = 0.3
        # subgroup discovery parameters
        self.ncutpoints = 20 # SD families
        self.beam_width = 100 # SD families
        self.sd_depth = 8 # SD families
        self.n_subgroups = 5


class Config_Real_World:
    def __init__(self):
        self.lr_flow = 5e-2
        self.lr_classifier = 2e-2
        self.alpha = 0.3
        self.lambd = 2.0
        self.pop_train_epochs = 1000
        self.subgroup_train_epochs = 1000
        self.final_fit_epochs = 0
        self.temperature = 0.2
        self.bin_deviation = 0.2
        self.use_weights = True
        # subgroup discovery parameters
        self.ncutpoints = 20 # SD families
        self.beam_width = 100 # SD families
        self.sd_depth = 8 # SD families
        def flow_gen():
            return bij.Spline(count_bins=12)
        self.flow_gen = flow_gen
        self.n_subgroups = 5



class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "process": record.process,
            "message": record.getMessage(),
        }

        # Include exception info if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record)


def setup_logger(
    name: str,
    root_dir: str,
    pid: int,
    log_dir: str = "logs",
    level: int = logging.INFO,
    max_bytes: int = 10_000_000,  # 10 mb
    backup_count: int = 5,
) -> logging.Logger:

    if not Path(root_dir).is_dir():
        raise ValueError(f"Root directory {root_dir} is not a valid directory")

    log_path = Path(f"{root_dir}/{log_dir}")
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    filename = f"{name}_{pid}.log"
    handler = RotatingFileHandler(
        log_path / filename,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )

    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

    logger.propagate = False  # Avoid duplicate root logging

    # logger.info("Rotating JSON log enabled")

    return logger