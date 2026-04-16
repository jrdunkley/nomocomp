# nomocomp

Exact geometric model comparison via fibre-volume correction.

**Documentation:** [docs.nomogenetics.com/nomocomp](https://docs.nomogenetics.com/nomocomp.html)

**Version:** 0.1.0

**Requires:** `nomogeo >= 0.4.0`, `numpy`, `statsmodels`

## What it does

nomocomp replaces AIC/BIC parameter-count penalties with the actual log-determinant of the observed Fisher information — the exact fibre-volume correction in the Gaussian-fibre sector.

For nested Gaussian linear models:
- AIC uses `-2 log L + 2k`
- BIC uses `-2 log L + k log(n)`
- nomocomp uses `-2 log L + log det(I) - k log(2pi)`

The third formula accounts for how tightly concentrated the posterior is in each parameter direction, not just how many parameters there are.

## Install

```bash
pip install nomocomp
```

## Flagship result

In nested Gaussian families with collinear predictors, the omitted determinant correction reverses both AIC and BIC rankings. A 56-cell collinearity phase diagram over 200 seeds shows the reversal activating continuously as correlation increases.

- At rho >= 0.9, n >= 60: exact Laplace evidence identifies the true model in 99-100% of seeds
- AIC identifies it in ~16%; BIC in ~6%
- The determinant correction is +1.5 to +3.9 nats depending on collinearity

When predictors are independent, AIC and BIC already work well and nomocomp agrees with them. The advantage appears where they fail.

## Quick start

```python
from nomocomp.extraction import extract_information
from nomocomp.comparator import GeometricModelComparator

# Extract Fisher information from fitted models
score = extract_information(ols_result, label="M1")

# Compare models with geometry
comp = GeometricModelComparator()
result = comp.compare({"M1": fit1, "M2": fit2})
print(result.geometric_ranking)
print(result.ranking_reversals)
```

## Modules

### `nomocomp.extraction`

Extracts the observed Fisher information matrix from fitted statsmodels results.

```python
score = extract_information(ols_result, label="M1")
# score.geometric_score    — exact Laplace score (lower = better)
# score.information_matrix — observed Fisher information at MLE
# score.log_det_information — log det(I)
# score.aic, score.bic     — standard criteria for comparison
```

### `nomocomp.comparator`

Compares multiple fitted models and detects ranking reversals.

```python
result = comp.compare({"M1": fit1, "M2": fit2})
# result.geometric_ranking — models ranked by exact score
# result.ranking_reversals — where geometric disagrees with AIC/BIC
# result.branch_reversal   — nomogeo branch-reversal detection
```

## Honest boundaries

- **Exact in the Gaussian-fibre sector.** For non-Gaussian likelihoods, the information matrix captures local curvature only — still sharper than a parameter count, but not the exact fibre volume.
- **Flat improper prior** on all parameters. For nested models the prior on shared parameters cancels.
- **"Exact local Laplace"** means exact to quadratic order at the MLE. Not the exact marginal likelihood.
- **Not yet a state-space comparator.** The state-space comparison pipeline is not yet built.

## Verification

```bash
python -m pytest tests/ -q   # 22 tests
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).
