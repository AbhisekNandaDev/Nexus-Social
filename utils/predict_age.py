from deepface import DeepFace
import numpy as np
import cv2
from utils.logger import get_logger

logger = get_logger(__name__)


def predict_age(image_input):
    logger.debug("predict_age called | input_type=%s", type(image_input).__name__)

    if isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        logger.debug("Decoded image from bytes | shape=%s", img.shape if img is not None else "None")
    else:
        img = image_input

    try:
        result = DeepFace.analyze(
            img_path=img,
            actions=['age'],
            enforce_detection=False
        )
        age = result[0]["age"]
        logger.info("Age prediction complete | age=%s", age)
        return age
    except Exception as exc:
        logger.error("Age prediction failed | error=%s", exc, exc_info=True)
        raise
