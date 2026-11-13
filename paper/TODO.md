# SO(3) paper — TODO

## Claims to establish
1. **Weight decay drives pruning; purity needs norm control (explicit or implicit).**
   Caveat (user, 2026-11-12): a small initialisation is an implicit regularisation
   (gradient flow from near zero is biased towards low-norm solutions, like the
   vanishing-multiplier limit of the constrained problem weight decay imposes), so
   small-init runs without weight decay do NOT show that weight decay is irrelevant
   to purity. Established so far (R9b): with wd = 0, small init gives 100% purity
   (standard init 80–90%) but no pruning (1544–1840 units vs 475): explicit wd is
   needed for the d³ count. Remaining: lazy regime (init ×10, wd = 0), where the
   implicit bias disappears — mixed neurons there would show purity needs norm
   control; then state and prove.
   *Lazy result (2026-11-12):* init ×3 and ×10, wd = 0: partial fit (0.3 and 0.8) with
   mixed neurons (purity 0, median 0.59 = init). Not converged; larger lr diverged.

2. **Purity mechanism (hypothesis).** For a z² unit, z = Σ_l a_l(G) + Σ_l b_l(H) + c with
   a_l, b_l the irrep-l parts. By Schur orthogonality the residual in output block l
   only correlates with the products a_l b_l, so the gradient flow of a unit decouples
   per irrep at leading order: the l-components (a_l, b_l, r_l) of a unit form a
   degree-3 homogeneous system, θ̇_l ∝ (signal_l)·θ_l², whose growth is
   "rich get richer" (finite-time blow-up, time ∝ 1/(signal_l · θ_l(0))). The block with
   the largest initial alignment runs away first; once it has fitted part of the
   residual, the other blocks of that unit stop receiving signal, and the cross-sector
   terms a_l b_{l'} (l ≠ l'), which only add error, are pushed down. This is the
   utility-maximisation step of AGF (Marchetti et al.), where single-irrep directions
   maximise utility. Predictions: purity appears without weight decay from small
   init (rich regime); from large init (lazy regime) and wd = 0 units stay mixed;
   wd restores purity by shrinking the mixed, unused components. Test: init scale
   {0.1, 1, 10} × wd {0, 1e-3}, L=3, purity and count over time.
3. **Weight decay favours Fourier features as Dirichlet-energy minimisers** (user's
   conjecture; possibly a separate paper): the Frobenius norm of the first layer bounds
   the Dirichlet energy of the embedding; irreps are Laplacian eigenfunctions. Show
   that the weight-decay objective restricted to the selection step is minimised by
   irrep-aligned (Fourier) features.
4. **Minimum-norm decomposition of the matmul tensor.** Numerically minimise the
   weight-decay objective Σ_j (‖r_j‖² + ‖a_j‖² + ‖b_j‖²) subject to
   Σ_j r_j ⊗ a_j ⊗ b_j = M_d (d×d matrix multiplication), for d = 3, 5, 7, 9: does the
   optimum use d³ (or d³ + d) units? Then a proposition: for a block-diagonal target,
   a unit mixing irreps is never optimal.
   **Result (2026-11-12):** the numerical minimum is exactly d³ equal units for
   d = 3, 5, 7, 9 (3 seeds each, `scripts/min_norm_decomposition.py`). To do: prove it
   (balanced weight decay ⇒ per-unit cost 3(‖r‖‖a‖‖b‖)^{2/3}); test whether the extra
   d units seen in trained networks for l ≥ 3 cancel the non-bilinear z² terms (add
   those terms to the objective, or ablate the extra units in a trained network).

## Experiments
- [x] Seeds, pixel basis, L ≤ 6 (night of 2026-11-11/12).
- [x] Vanishing initialisation with per-neuron scaled SGD (R10: order l=1,2,3,4 confirmed) (η_i ∝ ‖θ_i‖^{1−k}, k = 2;
      Marchetti et al.), dense snapshots: learning order of irreps (Hypothesis 2).
- [~] wd = 0 control and init × wd grid (claims 1–2): done except init 10.
- [ ] Larger L with the truncated delta (L ≥ 6 at width ≥ Σ d_l³), with an optimiser
      that escapes the plateau (loss rescaling; per-neuron scaled SGD).
- [ ] Dynamics figure: per-block error and purity vs epoch on one axis, several wd.

## Paper
- [x] Remove LayerNorm experiments and mentions.
- [ ] Figures with error bars (3 seeds) instead of tables.
- [ ] Schematic of the three ingredients.
- [ ] Later, once results have converged: restructure to 9 pages (main: encodings,
      selection, mechanism; appendix: GH target, availability-bound proof,
      normalisation, harmonics) and add a Related Work section.
