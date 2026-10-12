# Proximity to criticality d_b: formula sheet

Notes for `src/analysis/prox_crit/`, a Python port of Sooter et al.'s MATLAB toolkit. This sheet is meant for quick recognition of the formulas in the code; derivations are in the paper.

### AR model and stability
**Context:** All quantities are defined on AR(n) models fitted to the z-scored series.

$$
x_t = \sum_{k=1}^{n}\phi_k\,x_{t-k} + \sigma\,\varepsilon_t,\qquad
P(z) = 1-\sum_{k=1}^{n}\phi_k z^k
$$

The model is stable when every root of $P$ satisfies $|z|>1$. In code, `explosive` uses the threshold $\min|z| < 0.99$.

### Critical manifold $M_b$
**Context:** The model is critical of type $b$ when $z=1$ is a root of $P$ with multiplicity $b/2$. This gives linear constraints (`beta_lift`), so $M_b$ is affine with dimension $n-b/2$.

$$
\sum_k \phi_k = 1,\qquad \sum_k k^j\phi_k = 0\quad (j=1,\dots,b/2-1)
$$

Basis points used by `calc_dbv1`: $(1-z)^k$ with $k=b/2,\dots,n$, i.e. $\phi_t=\binom{k}{t}(-1)^{t+1}$.

### Spectra and KL-divergence rate
**Context:** $d_b$ is the smallest KL rate from the fitted model to any model in $M_b$.

$$
\hat\phi(\omega)=\sum_k\phi_k e^{-i\omega k},\qquad g(\omega)=|1-\hat\phi(\omega)|^2\quad(\text{spectrum}=\sigma^2/g)
$$

$$
\sigma_c^2=\frac{1}{2\pi}\int_{-\pi}^{\pi}\frac{g_c}{g}\,d\omega,\qquad
\mathrm{klr}=\frac{1}{4\pi}\int_{-\pi}^{\pi}\Big[\log\frac{\sigma_c^2 g}{g_c}+\frac{g_c}{\sigma_c^2 g}-1\Big]d\omega
$$

The fitted model's $\sigma$ is set to 1, so $\sigma_c$ is the ratio $\sigma_c/\sigma$ (FLAG 2). The integrand has an integrable log singularity at $\omega=0$, because $g_c(0)=0$ on $M_b$ (CHANGED 1 and 4).

The time-domain identity used in the tests: with $e=(1,-\phi_c)$ and $\Gamma$ the autocovariance matrix of the fitted model,

$$
\sigma_c^2=e^\top\Gamma e,\qquad \mathrm{klr}=\tfrac12\log(e^\top\Gamma e)
$$

### Units and error bar
$$
d_b\,[\text{bits/s}] = \frac{\log_2 e}{\Delta t}\,\min_{\phi_c\in M_b}\mathrm{klr},\qquad
\mathrm{sd}(d_b)=\sqrt{-\nabla d_b^\top H^{-1}\nabla d_b}
$$

Here $H$ is the Hessian of the profile log-likelihood $\ell(\phi)=-\tfrac T2\log\sum r_t^2$, and $\nabla d_b$ is obtained by forward differences (FLAG 13).

### AR fitting
- Yule–Walker: $\Gamma\phi=\gamma$ and $\sigma^2=1-\gamma^\top\phi$.
- Maximum likelihood: $\phi=(X^\top X)^{-1}X^\top Y$ and $H_{jk}=T\,(2B_jB_k/A^2-(X^\top X)_{jk}/A)$, with $A=\sum r^2$ and $B=X^\top r$.

**Source:**
- Sooter et al., *Defining and measuring proximity to criticality*, bioRxiv (2025), doi:10.1101/2025.08.03.668332

[Return to implementation](../../src/analysis/prox_crit/__init__.py) · [Flags](../../src/analysis/prox_crit/FLAGS.md)
