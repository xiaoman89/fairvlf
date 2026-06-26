"""Demographic Prompt Consistency (DPC) loss — the language-side fairness term.

Intuition: if whether a face is fake is determined by synthesis traces, then
merely mentioning race/gender in the question should not change the model's
score. DPC penalises the score shift between a neutral prompt q0 and the
demographic-mentioning / fairness-instructed prompts, but only beyond a small
tolerance epsilon (so genuinely demographic-correlated forensic cues are not
erased).

    L_DPC = E_x  sum_{q in {q_fair, q_attr}}  [ max(0, |s(x,q) - s(x,q0)| - eps) ]^2
"""

import torch


def dpc_loss(score_neutral: torch.Tensor,
             score_variants: torch.Tensor,
             epsilon: float = 0.1) -> torch.Tensor:
    """Margin-tolerant prompt consistency.

    Args:
        score_neutral: (B,) calibrated fake score under the neutral prompt q0.
        score_variants: (B, V) scores under V demographic variants
                        (e.g. q_fair, q_attr).
        epsilon: tolerance; deviations within +/- epsilon are not penalised.

    Returns:
        Scalar loss.
    """
    # |s(x,q) - s(x,q0)| for each variant, shape (B, V)
    deviation = (score_variants - score_neutral.unsqueeze(1)).abs()
    # Only penalise the part beyond the tolerance band.
    excess = torch.clamp(deviation - epsilon, min=0.0)
    return (excess ** 2).mean()
