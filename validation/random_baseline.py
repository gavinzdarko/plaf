"""Random-label baseline validation.

A well-designed membership inference attack should perform no better than
chance against a model trained on random labels, because such a model has
no real signal to memorise.  If AUC > 0.6 against the random model, the
attack methodology itself may be producing false positives.
"""

from __future__ import annotations

import numpy as np

from core.target import TargetModel
from core.membership_probe import MembershipProbe
from config import AttackConfig


def validate_random_baseline(
    target_random: TargetModel,
    data: np.ndarray,
    labels: np.ndarray,
    config: AttackConfig | None = None,
) -> dict:
    """Run membership probe against a random-label model and check AUC ≤ 0.6.

    Parameters
    ----------
    target_random : TargetModel
        Model trained on randomly-shuffled labels.
    data : np.ndarray
        Combined member + non-member feature matrix.
    labels : np.ndarray
        Ground-truth membership labels (1 = member, 0 = non-member).
    config : AttackConfig, optional
        Attack configuration; uses defaults if *None*.

    Returns
    -------
    dict with keys ``random_auc`` (float) and ``valid`` (bool).
    """
    if config is None:
        config = AttackConfig()

    probe = MembershipProbe(target_random, config)

    # Use the non-member portion for calibration
    non_member_mask = labels == 0
    if non_member_mask.sum() == 0:
        raise ValueError("labels must contain at least some non-members (0)")

    probe.calibrate_baseline(data[non_member_mask])
    result = probe.probe(data, labels)

    auc = result.auc

    if auc > 0.6:
        print(
            f"WARNING: Random-baseline AUC = {auc:.3f} (> 0.6). "
            "Attack methodology may be producing false positives!"
        )

    return {"random_auc": auc, "valid": auc <= 0.6}
