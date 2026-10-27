# Symmetrised consistency

The symmetrised forecaster of any model \(f\) is

\[
\bar f(X) = \frac{1}{8} \sum_{g \in G} g^{-1}\big(f(g(X))\big).
\]

\(G = \{R^a,\; MR^a : a = 0,1,2,3\}\). \(R^a\) is \(a\) quarter-turns anticlockwise, and its inverse is \(R^{-a}\). Each \(MR^a\) is its own inverse. The same \(g\) is applied to every frame, \(f\) predicts, and \(g^{-1}\) is applied to every predicted frame.

\(C_M = C_R = 0\) for every \(f\), including a model that was never trained on mirrored or rotated clips.

Each \(g \in G\) permutes pixels in the same way on every frame, so each \(g\) is linear, and \(G\) is a group. Fix any \(h \in G\). Right multiplication by \(h\) is a bijection of \(G\): as \(g\) runs through \(G\), so does \(g' = gh\), and \(g^{-1} = h\,(g')^{-1}\). Therefore

\[
\bar f(h(X))
  = \frac{1}{8} \sum_g g^{-1}\big(f(gh(X))\big)
  = \frac{1}{8} \sum_{g'} h\big((g')^{-1}(f(g'(X)))\big)
  = h\big(\bar f(X)\big).
\]

The last step moves \(h\) through the average, which linearity allows. Nothing in the argument uses the weights or the architecture of \(f\). It only needs \(f\) to map a clip to a clip of the same shape, so that \(G\) can act on the output.

\(M\) and \(R\) are elements of \(G\): \(M = MR^0\) and \(R = R^1\). So \(\bar f(M(X)) = M(\bar f(X))\) and \(\bar f(R(X)) = R(\bar f(X))\). The consistency gaps are the mean squared differences of those pairs, and both differences are identically zero. Hence \(C_M = 0\) and \(C_R = 0\) exactly.

On the 1,000 test videos (indices 9000–9999) the measured gaps are floating-point residuals of that identity. Scientific notation, from `results/table6_bd.json`:

| Wrapped model | Seed | \(C_M\) | \(C_R\) |
| --- | --- | --- | --- |
| B | 0 | 8.131422e-17 | 6.487641e-17 |
| B | 1 | 8.724459e-17 | 7.022728e-17 |
| B | mean | 8.427941e-17 | 6.755185e-17 |
| D | 0 | 4.770134e-17 | 3.688488e-17 |
| D | 1 | 6.168978e-17 | 4.838336e-17 |
| D | mean | 5.469556e-17 | 4.263412e-17 |
