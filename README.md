# Mafinrisk-Thesis

The aim of this thesis is to estimate the European call price surface and the state price density.
The estimation of the price surface is both a fundamental goal of financial mathematics and
an arduous task to perform, due to dissimilarity of the strikes and maturities available, the
high bid-ask spread and the common presence of arbitrage opportunities. In this thesis, I will
discuss about a new estimation method that uses the Bernstein polynomial as a basis. The
major advantage of this approach is that it allows to cast the estimation problem as a quadratic
programming problem, where the entire surface is expressed as a convex combination of low
degree Bernstein polynomials. This allows to solve the problem sequentially, imposing only a
limited number of strike and calendar constraints. An efficiency study between the estimation
with and without calendar constraints is also conducted using the Heston model. Finally, this
method is tested on two different empirical datasets to verify its practical implementation and
strengths.
