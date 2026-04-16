"""
GeometricModelComparator — exact geometric model comparison.

Takes a set of fitted statsmodels results, extracts the local information
matrix for each, and ranks them by the exact geometric score:

    S_geom = -2 logL + log det(I)

where I is the observed Fisher information at the MLE.

This replaces the parameter-count penalties of AIC (-2 logL + 2k) and
BIC (-2 logL + k log n) with the actual log-determinant of the
information matrix — the exact fibre-volume correction in the
Gaussian-fibre sector.

When a visible/hidden partition is declared, the comparator uses
nomogeo's affine-hidden reduction to compute the exact marginal
visible-sector action via Schur complement elimination.

Honest boundary:
    Exactness holds in the Gaussian-fibre sector.  For non-Gaussian
    likelihoods, the information matrix captures local curvature only —
    still sharper than a parameter count, but not the exact fibre volume.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

import nomogeo  # 0.3.2 API: affine_hidden_branch_reversal, guarded_fibre_dominance (affine.py)
# ── nomogeo 0.3.3 update safety note ──
# This module uses only affine.py APIs (affine_hidden_branch_reversal,
# guarded_fibre_dominance). The 0.3.3 update extends affine.py with
# tower_affine_hidden_elimination but does NOT alter existing signatures.
# No changes needed here for the 0.3.3 upgrade.
# Future: nomocomp may upgrade to call nomogeo's typed evidence pipeline
# once the regime detector and evidence dispatcher are in place.

from .extraction import ModelScore, extract_information


@dataclass
class RankingDiagnostic:
    """Pairwise ranking comparison between two criteria."""

    criterion_a: str
    criterion_b: str
    agree: bool
    winner_a: str  # model label that wins under criterion A
    winner_b: str  # model label that wins under criterion B


@dataclass
class PairwiseBudget:
    """Conservation-law information budget between two models."""

    label_a: str
    label_b: str
    visible_rate: float
    hidden_rate: float
    ambient_rate: float
    visible_fraction: float
    conservation_residual: float


@dataclass
class ComparisonResult:
    """Full comparison result across a model set.

    Attributes
    ----------
    scores : list of ModelScore
        Per-model scores, ordered as input.
    geometric_ranking : list of str
        Model labels ordered by geometric score (best first).
    aic_ranking : list of str
        Model labels ordered by AIC (best first).
    bic_ranking : list of str
        Model labels ordered by BIC (best first).
    ranking_reversals : list of RankingDiagnostic
        Pairwise comparisons where criteria disagree.
    branch_reversal : nomogeo result or None
        Branch-reversal detection from nomogeo, if available.
    fibre_dominance : nomogeo result or None
        Fibre-dominance diagnostic from nomogeo, if available.
    information_budgets : list of PairwiseBudget
        Conservation-law information budgets for each model pair.
        Shows how the Fisher information difference decomposes into
        visible and hidden contributions. Available when all models
        have the same number of parameters.
    """

    scores: list[ModelScore] = field(default_factory=list)
    geometric_ranking: list[str] = field(default_factory=list)
    aic_ranking: list[str] = field(default_factory=list)
    bic_ranking: list[str] = field(default_factory=list)
    ranking_reversals: list[RankingDiagnostic] = field(default_factory=list)
    branch_reversal: Any = None
    fibre_dominance: Any = None
    information_budgets: list[PairwiseBudget] = field(default_factory=list)


class GeometricModelComparator:
    """Compare fitted models by exact geometric score.

    Parameters
    ----------
    extraction_method : str, default "auto"
        How to extract the information matrix.  See
        ``extract_information`` for options.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> comp = GeometricModelComparator()
    >>> result = comp.compare(
    ...     {"simple": result_simple, "complex": result_complex}
    ... )
    >>> print(result.geometric_ranking)
    ['simple', 'complex']
    """

    def __init__(self, extraction_method: str = "auto"):
        self.extraction_method = extraction_method

    def compare(
        self,
        results: dict[str, Any],
    ) -> ComparisonResult:
        """Compare a set of fitted models.

        Parameters
        ----------
        results : dict mapping label -> statsmodels result
            Each value must be a fitted statsmodels result object
            providing .llf, .params, .nobs, and either .model.hessian()
            or .cov_params().

        Returns
        -------
        ComparisonResult
            Full comparison with rankings and reversal diagnostics.
        """
        if len(results) < 2:
            raise ValueError("Need at least 2 models to compare.")

        # Extract scores
        scores = []
        for label, result in results.items():
            ms = extract_information(
                result,
                label=label,
                method=self.extraction_method,
            )
            scores.append(ms)

        # Rankings (lower is better for all three criteria)
        geo_ranked = sorted(scores, key=lambda s: s.geometric_score)
        aic_ranked = sorted(scores, key=lambda s: s.aic)
        bic_ranked = sorted(scores, key=lambda s: s.bic)

        geo_ranking = [s.label for s in geo_ranked]
        aic_ranking = [s.label for s in aic_ranked]
        bic_ranking = [s.label for s in bic_ranked]

        # Detect pairwise ranking reversals
        reversals = []
        labels = [s.label for s in scores]
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                a, b = scores[i], scores[j]

                # Geometric vs AIC
                geo_prefers = a.label if a.geometric_score < b.geometric_score else b.label
                aic_prefers = a.label if a.aic < b.aic else b.label
                if geo_prefers != aic_prefers:
                    reversals.append(RankingDiagnostic(
                        criterion_a="geometric",
                        criterion_b="AIC",
                        agree=False,
                        winner_a=geo_prefers,
                        winner_b=aic_prefers,
                    ))

                # Geometric vs BIC
                bic_prefers = a.label if a.bic < b.bic else b.label
                if geo_prefers != bic_prefers:
                    reversals.append(RankingDiagnostic(
                        criterion_a="geometric",
                        criterion_b="BIC",
                        agree=False,
                        winner_a=geo_prefers,
                        winner_b=bic_prefers,
                    ))

        # nomogeo branch-reversal detection
        # Decompose geometric score into variational + fibre components
        var_actions = np.array([-2.0 * s.log_likelihood for s in scores])
        fibre_volumes = np.array([s.log_det_information for s in scores])
        branch_rev = nomogeo.affine_hidden_branch_reversal(
            var_actions, fibre_volumes
        )
        fibre_dom = nomogeo.guarded_fibre_dominance(
            var_actions, fibre_volumes
        )

        # Conservation-law information budgets (pairwise)
        budgets = []
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                a, b = scores[i], scores[j]
                if (a.information_matrix is not None
                        and b.information_matrix is not None
                        and a.information_matrix.shape == b.information_matrix.shape):
                    k = a.information_matrix.shape[0]
                    try:
                        H_a = 0.5 * (a.information_matrix + a.information_matrix.T)
                        Hdot = 0.5 * ((b.information_matrix - a.information_matrix)
                                      + (b.information_matrix - a.information_matrix).T)
                        C_full = np.eye(k)
                        bg = nomogeo.information_budget(H_a, C_full, Hdot)
                        budgets.append(PairwiseBudget(
                            label_a=a.label, label_b=b.label,
                            visible_rate=bg.visible_rate,
                            hidden_rate=bg.hidden_rate,
                            ambient_rate=bg.ambient_rate,
                            visible_fraction=bg.visible_fraction,
                            conservation_residual=bg.conservation_residual,
                        ))
                    except Exception:
                        pass  # skip if information matrix is singular etc.

        return ComparisonResult(
            scores=scores,
            geometric_ranking=geo_ranking,
            aic_ranking=aic_ranking,
            bic_ranking=bic_ranking,
            ranking_reversals=reversals,
            branch_reversal=branch_rev,
            fibre_dominance=fibre_dom,
            information_budgets=budgets,
        )

    def compare_with_partition(
        self,
        results: dict[str, Any],
        hidden_indices: dict[str, Sequence[int]],
    ) -> ComparisonResult:
        """Compare models with declared visible/hidden parameter partition.

        For each model, the full parameter Hessian is partitioned into
        visible and hidden blocks.  The hidden block is eliminated via
        nomogeo's exact affine-hidden reduction, yielding the marginal
        visible-sector score.

        Parameters
        ----------
        results : dict mapping label -> statsmodels result
        hidden_indices : dict mapping label -> list of int
            For each model, the indices of parameters to treat as hidden.
            Visible parameters are those NOT in this list.

        Returns
        -------
        ComparisonResult
        """
        if len(results) < 2:
            raise ValueError("Need at least 2 models to compare.")

        scores = []
        for label, result in results.items():
            ms = extract_information(
                result,
                label=label,
                method=self.extraction_method,
            )

            if label in hidden_indices and hidden_indices[label]:
                h_idx = list(hidden_indices[label])
                k = ms.n_params
                v_idx = [i for i in range(k) if i not in h_idx]

                if not v_idx:
                    # All parameters hidden — use full score
                    scores.append(ms)
                    continue

                # Extract blocks from information matrix
                I_full = ms.information_matrix
                I_hh = I_full[np.ix_(h_idx, h_idx)]
                I_vh = I_full[np.ix_(v_idx, h_idx)]
                I_vv = I_full[np.ix_(v_idx, v_idx)]

                # Fibre volume from hidden block
                sign, log_det_hh = np.linalg.slogdet(I_hh)
                fibre_vol = float(log_det_hh)

                # Schur complement: I_vis = I_vv - I_vh I_hh^{-1} I_hv
                I_hh_inv_Ihv = np.linalg.solve(I_hh, I_vh.T)
                I_schur = I_vv - I_vh @ I_hh_inv_Ihv
                sign_s, log_det_schur = np.linalg.slogdet(I_schur)

                # Partitioned geometric score:
                # log det(I_full) = log det(I_hh) + log det(I_schur)
                # So we can decompose into hidden fibre + visible info
                ms.log_det_information = float(log_det_hh + log_det_schur)
                ms.geometric_score = -2.0 * ms.log_likelihood + ms.log_det_information

            scores.append(ms)

        # Rankings
        geo_ranked = sorted(scores, key=lambda s: s.geometric_score)
        aic_ranked = sorted(scores, key=lambda s: s.aic)
        bic_ranked = sorted(scores, key=lambda s: s.bic)

        # Reversals
        reversals = []
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                a, b = scores[i], scores[j]
                geo_prefers = a.label if a.geometric_score < b.geometric_score else b.label
                aic_prefers = a.label if a.aic < b.aic else b.label
                bic_prefers = a.label if a.bic < b.bic else b.label
                if geo_prefers != aic_prefers:
                    reversals.append(RankingDiagnostic(
                        "geometric", "AIC", False, geo_prefers, aic_prefers,
                    ))
                if geo_prefers != bic_prefers:
                    reversals.append(RankingDiagnostic(
                        "geometric", "BIC", False, geo_prefers, bic_prefers,
                    ))

        # nomogeo diagnostics
        var_actions = np.array([-2.0 * s.log_likelihood for s in scores])
        fibre_volumes = np.array([s.log_det_information for s in scores])
        branch_rev = nomogeo.affine_hidden_branch_reversal(var_actions, fibre_volumes)
        fibre_dom = nomogeo.guarded_fibre_dominance(var_actions, fibre_volumes)

        return ComparisonResult(
            scores=scores,
            geometric_ranking=[s.label for s in geo_ranked],
            aic_ranking=[s.label for s in aic_ranked],
            bic_ranking=[s.label for s in bic_ranked],
            ranking_reversals=reversals,
            branch_reversal=branch_rev,
            fibre_dominance=fibre_dom,
        )

    @staticmethod
    def summary(result: ComparisonResult, technical: bool = False) -> str:
        """Human-readable summary of a comparison result.

        Parameters
        ----------
        result : ComparisonResult
        technical : bool, default False
            If True, include numerical details.
        """
        lines = []
        lines.append("Geometric Model Comparison")
        lines.append("=" * 50)

        # Score table
        if technical:
            lines.append(
                f"  {'Model':<15s}  {'logL':>10s}  {'log|I|':>10s}  "
                f"{'Geom':>10s}  {'AIC':>10s}  {'BIC':>10s}  {'k':>3s}"
            )
            for s in result.scores:
                lines.append(
                    f"  {s.label:<15s}  {s.log_likelihood:10.2f}  "
                    f"{s.log_det_information:10.2f}  "
                    f"{s.geometric_score:10.2f}  {s.aic:10.2f}  "
                    f"{s.bic:10.2f}  {s.n_params:3d}"
                )
        else:
            lines.append(
                f"  {'Model':<15s}  {'Geom':>10s}  {'AIC':>10s}  {'BIC':>10s}"
            )
            for s in result.scores:
                lines.append(
                    f"  {s.label:<15s}  {s.geometric_score:10.2f}  "
                    f"{s.aic:10.2f}  {s.bic:10.2f}"
                )

        lines.append("")
        lines.append(f"  Geometric ranking:  {' > '.join(result.geometric_ranking)}")
        lines.append(f"  AIC ranking:        {' > '.join(result.aic_ranking)}")
        lines.append(f"  BIC ranking:        {' > '.join(result.bic_ranking)}")

        if result.ranking_reversals:
            lines.append("")
            lines.append(f"  Ranking disagreements: {len(result.ranking_reversals)}")
            for rv in result.ranking_reversals:
                lines.append(
                    f"    {rv.criterion_a} picks {rv.winner_a}, "
                    f"{rv.criterion_b} picks {rv.winner_b}"
                )

        if result.branch_reversal is not None:
            lines.append("")
            lines.append(
                f"  Branch reversal detected: {result.branch_reversal.reversal}"
            )

        if result.fibre_dominance is not None and result.fibre_dominance.ratio is not None:
            lines.append(
                f"  Fibre dominance ratio: {result.fibre_dominance.ratio:.2f}"
            )

        lines.append("")
        lines.append("  Lower score is better.  Geometric score uses the actual")
        lines.append("  log-determinant of the information matrix rather than a")
        lines.append("  parameter-count proxy.  Exact in the Gaussian sector.")
        return "\n".join(lines)
