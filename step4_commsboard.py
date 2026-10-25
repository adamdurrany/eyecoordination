
"""EyeComm communication board, with caregiver panel off by default."""
from step5_dualsystem import run_app


if __name__ == "__main__":
    run_app(show_caregiver=False)
