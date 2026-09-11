# config.py
"""
CALIBRATED_THRESHOLD starts as a placeholder and gets overwritten by
main.py's startup event before the first request is served.
"""

CALIBRATED_THRESHOLD = 0.5  # placeholder, overwritten at startup


def set_calibrated_threshold(value: float) -> None:
    global CALIBRATED_THRESHOLD
    CALIBRATED_THRESHOLD = value
