from typing import Dict, Tuple

import numpy as np


def t_noise(n: int, df: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Generate Student-t noise scaled by `sigma`.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    df : int, optional
        Degrees of freedom for the Student-t distribution (default: 5).
    sigma : float, optional
        Scale multiplier applied to the sampled t variates (default: 1.0).

    Returns
    -------
    numpy.ndarray
        1-D array of length `n` containing scaled Student-t random samples.
    """

    return np.random.standard_t(df, size=n) * sigma


def generate_scenarios() -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Return a set of synthetic signals and their noiseless versions.

    The function produces several named scenarios (arrays) and returns a
    tuple of two dictionaries: (noisy_signals, true_signals). Each dict maps
    a human-readable name (e.g. "Scenario 1") to a 1-D numpy array.

    Returns
    -------
    (dict, dict)
        (noisy_signals, true_signals)
    """

    np.random.seed(42)

    # scenario 1
    n1 = 2048
    pos = np.array([0.1, 0.13, 0.15, 0.23, 0.25, 0.40, 0.44, 0.65, 0.76, 0.78, 0.81])
    h = np.array([4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2])
    sig1 = np.zeros(n1)
    for p, hi in zip(pos, h):
        sig1[int(p * n1) :] += hi
    sig1 = sig1 * 3.5
    y1 = sig1 + t_noise(n1, df=5, sigma=5.0)

    # Scenario 2
    n2 = 512
    sig2 = np.zeros(n2)
    segments_fms = [
        (0, 200, -0.2),
        (200, 220, 0.2),
        (220, 240, 1.2),
        (240, 260, -0.8),
        (260, 280, 0.5),
        (280, 400, -0.1),
        (400, 420, 0.8),
        (420, 512, -0.3),
    ]
    for start, end, val in segments_fms:
        sig2[start:end] = val
    y2 = sig2 + t_noise(n2, df=5, sigma=0.3)

    y2_prime = sig2 + t_noise(n2, df=5, sigma=0.2)

    # Scenario 3: short and long segments
    n3 = 512
    sig3 = np.zeros(n3)
    curr = 0
    while curr < n3:
        length = np.random.choice([10, 20, 80, 100])
        val = np.random.normal(0, 2)
        end = min(curr + length, n3)
        sig3[curr:end] = val
        curr = end

    y3 = sig3 + t_noise(n3, df=5, sigma=0.3)

    # Scenario 4: saw tooth
    n4 = 240
    sig4 = np.zeros(n4)
    n_teeth = 10
    len_tooth = n4 // n_teeth
    for i in range(n_teeth):
        if i % 2 == 1:
            sig4[i * len_tooth : (i + 1) * len_tooth] = 1.0
    y4 = sig4 + t_noise(n4, df=5, sigma=0.3)

    # Scenario 5: Stairs
    n5 = 150
    sig5 = np.zeros(n5)
    n_steps = 10
    len_step = n5 // n_steps
    for i in range(n_steps):
        sig5[i * len_step : (i + 1) * len_step] = (i + 1) * 1.5
    y5 = sig5 + t_noise(n5, df=5, sigma=0.3)

    noisy = {
        "Scenario 1": y1,
        "Scenario 2": y2,
        "Scenario 2'": y2_prime,
        "Scenario 3": y3,
        "Scenario 4": y4,
        "Scenario 5": y5,
    }

    truth = {
        "Scenario 1": sig1,
        "Scenario 2": sig2,
        "Scenario 2'": sig2,
        "Scenario 3": sig3,
        "Scenario 4": sig4,
        "Scenario 5": sig5,
    }

    return noisy, truth


if __name__ == "__main__":
    _, _ = generate_scenarios()
