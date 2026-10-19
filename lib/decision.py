"""Turning predicted returns into studio profit, and simulating whole slates of films."""

import numpy as np

# the studio keeps half the box office and spends another quarter of the budget on marketing,
# which puts break-even at 2.5 times the budget, the same line as the flop definition
STUDIO_SHARE = 0.5
MARKETING_SHARE = 0.25


def simulate_slate_profit(
    mean_log_multiple: np.ndarray,
    sd_log_multiple: np.ndarray,
    budget_musd: np.ndarray,
    standardised_errors: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Studio profit of a slate in each of `draws` simulated years, USD millions.

    The first three arrays hold one value per film in the slate. Each film's log multiple is its mean plus its
    standard deviation times an error drawn on its own from `standardised_errors`, and its profit is
    STUDIO_SHARE * gross - (1 + MARKETING_SHARE) * budget.
    """
    errors = rng.choice(standardised_errors, size=(draws, len(budget_musd)))
    log_multiple = mean_log_multiple + sd_log_multiple * errors
    gross = np.exp(log_multiple) * budget_musd
    profit = STUDIO_SHARE * gross - (1 + MARKETING_SHARE) * budget_musd
    return profit.sum(axis=1)
