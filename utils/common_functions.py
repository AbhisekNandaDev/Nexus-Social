import yaml
from utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path):
    logger.debug("Loading config | path=%s", config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info("Config loaded | path=%s keys=%s", config_path, list(config.keys()) if config else [])
    return config
