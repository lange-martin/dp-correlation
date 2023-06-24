# Differential Privacy, Correlation, and the Associative Privacy Loss

This python script contains experiments that measure the associative privacy loss 
for a differentially private query.

The associative privacy loss is defined the following way:

Let $x_i \in \mathcal{X}$ be a record from the universe of records.
Let the random variable $X_i$ model the value of the $i$-th record 
of the dataset $D \in \mathcal{X}^n$, 
and let the random variable $Y$ model the output of algorithm $\mathcal{A}$. 
The associative privacy loss is
$$\mathcal{L}(S) = \sup_{x_i, x_i' \in \mathcal{X}} \log \frac{Pr[Y \in S | X_i = x_i]}{Pr[Y \in S | X_i = x_i']}$$