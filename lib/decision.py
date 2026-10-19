"""Turning predicted box office into studio profit, and simulating whole slates of films."""

import numpy as np

# the studio keeps half the box office and spends another quarter of the budget on marketing,
# which puts break-even at 2.5 times the budget, the same line as the flop definition
STUDIO_SHARE = 0.5
MARKETING_SHARE = 0.25


def simulate_slate_profit(
    predicted_log_gross: np.ndarray,
    budget_musd: np.ndarray,
    film_errors: list[np.ndarray],
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Studio profit of a slate in each of `draws` simulated years, USD millions.

    The two arrays and the list hold one entry per film in the slate. Each film's log gross is its prediction
    plus an error drawn on its own from its entry in `film_errors` (the training errors of its budget tier),
    and its profit is STUDIO_SHARE * gross - (1 + MARKETING_SHARE) * budget.
    """
    errors = np.column_stack(
        [rng.choice(film_error, size=draws) for film_error in film_errors]
    )
    gross = np.exp(predicted_log_gross + errors)
    profit = STUDIO_SHARE * gross - (1 + MARKETING_SHARE) * budget_musd
    return profit.sum(axis=1)
