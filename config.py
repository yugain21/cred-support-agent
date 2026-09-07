# config.py
"""
Shared runtime config. CALIBRATED_THRESHOLD starts at a conservative
placeholder and MUST be overwritten by calling `set_calibrated_threshold()`
after running the empirical calibration in eval.py / main.py startup.
Any code that imports CALIBRATED_THRESHOLD before calibration runs will
get this placeholder - main.py's startup event runs calibration before
any request is served, so this is safe in the deployed app.
"""

CALIBRATED_THRESHOLD = 0.5  # placeholder - overwritten at startup


def set_calibrated_threshold(value: float) -> None:
    global CALIBRATED_THRESHOLD
    CALIBRATED_THRESHOLD = value
