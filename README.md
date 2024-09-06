# Mafinrisk-Thesis

The aim of this thesis is to estimate the European call price surface and the state price den
sity. The price surface estimation is both a fundamental goal of financial mathematics and a
challenging task to perform, due to the differences in the strike prices across available matu
rities, the wide bid-ask spread and the frequent presence of arbitrage opportunities. In this
thesis, I will discuss a new estimation method that uses the Bernstein polynomial as the basis.
The major advantage of this approach is that it allows the estimation problem to be cast as a
quadratic programming problem, where the entire surface is expressed as a convex combination
of low-degree Bernstein polynomials. This allows the problem to be solved sequentially, impos
ing only a limited number of strike and calendar constraints. An efficiency study comparing the
estimation with and without calendar constraints is also conducted using the Heston model.
Finally, this method is tested on two empirical datasets to verify its practical implementation
and strengths.

