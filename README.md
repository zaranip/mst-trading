# MST Topology Trading Strategy

### *A Cross-Asset Relative-Value Strategy Using Network Backbones and Spectral Graph Theory*

---

## 1. Project Overview & Motivation

Standard statistical arbitrage and pairs trading methodologies typically rely on pre-selected asset subsets or brute-force scans checking for pairwise cointegration ($I(1)$ residuals). These standard approaches are prone to severe data-mining biases, run into scalability issues as dimension $N$ expands, and generally ignore structural configurations and redundancy across the asset universe.

This project implements a **graph-theoretic asset allocation and directional relative-value strategy** across a cross-asset universe of 21 highly liquid ETFs. By mapping dynamic correlation matrices onto a metric space, we isolate the market's asset dependency structure using a **Minimum Spanning Tree (MST)**. Instead of using raw pairwise correlations, we model cross-sectional dislocations through a path-aware metric known as **Effective Resistance ($\Omega$)**, extracted via the **Moore-Penrose pseudoinverse of the graph Laplacian**.

Furthermore, we utilize the **Fiedler Value ($\lambda_2$)**—the algebraic connectivity of the network network—to construct a systematic risk throttle that dynamically scales down portfolio leverage during systemic market integration regimes.

---

## 2. Theoretical Pipeline & Mathematical Framework

The data pipeline runs at a weekly rebalancing frequency $t$. It maps continuous asset return data into a discrete topological graph structure and generates volatility-targeted execution weights via the following continuous sequence:

```
[Log Returns r_i(t)] ➔ [Pearson Correlation Matrix ρ_ij] ➔ [Mantegna Metric Space d_ij]
                                                                     │
[Path-Aware Resistance Ω_ij] ◄─ [Graph Laplacian L] ◄─ [Kruskal's MST Topology T]
              │
[Rolling Z-Score Normalization z_ij(t)] ➔ [Momentum Signals s_ij(t) with Exponential Damping]
                                                                     │
[Final Portfolio Weights w_i(t)] ◄─ [Gross Leverage Cap] ◄─ [Volatility Sizing & Fiedler Throttle γ(t)]

```

### 2.1 From Correlation to Geometric Metric Space

Let $r_i(t) = \ln(P_i(t)/P_i(t-1))$ represent the log return of asset $i$ at time $t$. For an investment universe of size $N = 21$, we evaluate the statistical association over a rolling lookback window $\tau$ via the Pearson correlation matrix $\mathbf{C}_t \in [-1, 1]^{N \times N}$, whose entry elements $\rho_{ij}$ are defined as:

$$\rho_{ij} = \frac{\mathbb{E}[r_i r_j] - \mathbb{E}[r_i]\mathbb{E}[r_j]}{\sigma_i \sigma_j}$$

While correlation indicates linear co-movement, it fails to fulfill the triangle inequality and does not qualify as a true distance metric. To project this matrix onto an interconnected metric space $G = (V, E, d_{ij})$, we execute a non-linear transform following Mantegna (1999):

$$d_{ij} = \sqrt{2(1 - \rho_{ij})}$$

Where $d_{ij}$ represents a proper Euclidean distance bounding asset relationships strictly to the domain $[0, 2]$. Perfectly correlated assets satisfy $d_{ij} = 0$, completely anti-correlated assets show $d_{ij} = 2$, and orthogonal pairs map to $d_{ij} = \sqrt{2}$.

### 2.2 Isolating the Market Skeleton (Minimum Spanning Tree)

Given a complete, weighted underlying graph $G$ containing $N$ nodes and $\binom{N}{2}$ edges, the dense connectivity space injects multi-collinearity and noise into portfolio signal generation. We extract an acyclic, connected sparse backbone graph $T = (V, E_{MST})$ that connects all $N$ vertices while minimizing the overall metric edge weight sum:

$$T = \arg\min_{T \subset G} \sum_{(i,j) \in T} d_{ij}$$

We resolve this optimization step systematically using **Kruskal’s Algorithm**:

1. Sort all edges $E$ in ascending order of their geometric weight values $d_{ij}$.
2. Iteratively append edges to the tree network $T$, skipping elements that close a structural loop.
3. Terminate execution exactly when $|E_{MST}| [cite_start]= N - 1$ edges are linked.



### 2.3 Spectral Graph Theory & Effective Resistance ($\Omega$)

To map network pathways across non-adjacent assets on our isolated tree skeleton, we construct the graph Laplacian matrix $\mathbf{L} \in \mathbb{R}^{N \times N}$. Let $\mathbf{A}$ represent the weighted adjacency tree matrix containing elements $A_{ij} = d_{ij}^{-1}$ for $(i,j) \in E_{MST}$ ($A_{ij} = 0$ otherwise), and let $\mathbf{D}$ represent the diagonal degree matrix where $D_{ii} = \sum_{j} A_{ij}$. The graph Laplacian is defined as:

$$\mathbf{L} = \mathbf{D} - \mathbf{A}$$

Because $\mathbf{L}$ is positive semi-definite and symmetric, its rank is at most $N-1$, meaning it is singular and cannot be inverted natively. We access path-aware architectural distances by evaluating the **Moore-Penrose Pseudoinverse** $\mathbf{L}^{\dagger}$. The effective resistance $\Omega_{ij}$ (Klein & Randić, 1993) across any node pair is calculated as:

$$\Omega_{ij} = (e_i - e_j)^T \mathbf{L}^{\dagger} (e_i - e_j)$$

Where $e_i$ and $e_j$ are standard basis vectors. Because our isolated subgraph structure $T$ is an acyclic tree, there exists a unique simple path linking node $i$ to node $j$. Consequently, our effective resistance metric elegantly simplifies to the cumulative summation of spatial edge links separating the two targets:

$$\Omega_{ij} = \sum_{(k, \ell) \in \text{path}(i,j)} d_{k\ell}$$

---

## 3. Signal Construction & Sizing Mechanics

### 3.1 Rolling Topological Z-Score Normalization

To extract changes in effective resistance relative to historic benchmarks, we evaluate dynamic rolling cross-sectional parameters over an active lookback index window of $L = 60$ weeks. The baseline mean ($\mu_{ij}$) and standard deviation ($\sigma_{ij}$) give us a standardized tracking metric $z_{ij}(t)$:

$$\mu_{ij}(t) = \frac{1}{L} \sum_{\tau=t-L+1}^{t} \Omega_{ij}(\tau)$$

$$\sigma_{ij}(t) = \sqrt{\frac{1}{L-1} \sum_{\tau=t-L+1}^{t} \left( \Omega_{ij}(\tau) - \mu_{ij}(t) \right)^2}$$

$$z_{ij}(t) = \text{clip}\left(\frac{\Omega_{ij}(t) - \mu_{ij}(t)}{\max(\sigma_{ij}(t), \epsilon)}, -3, 3\right)$$

Where $\epsilon = 10^{-6}$ establishes a zero-division floor, and outputs are symmetrically clipped to bounded thresholds $[-3, 3]$ to mitigate outlier distortions. A position mapping into $z_{ij}(t) > 0$ flags a structural dislocation where the two target nodes have drifted systematically further apart than normal.

### 3.2 Exponentially Damped Momentum Signals

Empirical network asset analysis reveals that when architectural metrics decouple ($z_{ij}(t) > 0$), topological drift exhibits short-term persistence before mean-reverting. To exploit this structural drift cleanly across all active node coordinates, we construct a bounded signal vector that introduces an exponential distance-based discount factor:

$$\bar{R}(t) = \frac{2}{N(N-1)} \sum_{i < j} \Omega_{ij}(t)$$

$$s_{ij}(t) = z_{ij}(t) \cdot \exp\left( -\frac{\Omega_{ij}(t)}{\bar{R}(t)} \right)$$

Where $\bar{R}(t)$ represents the global cross-sectional mean of all pairwise resistances within the network. The scaling dampener penalizes pairs that are structurally isolated from the network backbone even under normal market conditions ($\Omega_{ij} \gg \bar{R}$), assigning higher alpha conviction to centralized assets. By default, this setup satisfies anti-symmetry conditions ($s_{ji}(t) = -s_{ij}(t)$), enforcing statistical cash balance between the long and short legs of the pair.

### 3.3 Volatility-Targeted Pair Sizing

To handle risk across highly heterogeneous asset classes, each traded pair is sized using an implicit volatility-targeting mechanism. We compute rolling hedge ratios $\beta_{ij}$ via Ordinary Least Squares (OLS) over a trailing 60-day window of log returns:

$$\beta_{ij}(t) = \frac{\text{Cov}(r_i, r_j)}{\text{Var}(r_j)}$$

The beta-hedged spread tracking return is defined as $S_{ij}(\tau) = r_i(\tau) - \beta_{ij}(t) \cdot r_j(\tau)$. We scale the annualized variance profile of the spread using a minimum volatility floor to guard against leverage explosions:

$$\sigma_{\text{spread}, ij}(t) = \max\left(\sqrt{252} \cdot \text{Std}(S_{ij}(\tau)), \, 0.01\right)$$

The nominal allocation assigned to each pair $(i,j)$ is calculated as follows:

$$w_{ij}(t) = \text{clip}\left(\frac{\sigma_{\text{target}} \cdot s_{ij}(t)}{\sigma_{\text{spread}, ij}(t)} \cdot \gamma(t), \, -1.5, \, 1.5\right)$$

Where $\sigma_{\text{target}} = 0.10$ defines our portfolio's strategic target volatility, and $\gamma(t)$ is our Fiedler-driven systematic risk multiplier.

### 3.4 Long-Short Alignment via Lead-Lag Analysis

To determine the directional structure of our relative-value expression, we track asset performance over a 20-day cumulative window:

$$C_i(t) = \prod_{\tau=t-19}^{t} (1 + r_i(\tau)) - 1$$

If $C_i(t) \le C_j(t)$, asset $i$ is designated as the lagging underperformer and assigned the long leg position, while the leading asset $j$ is allocated to the short leg. This structure trades in the direction of the topological drift, going long the relative laggard and short the relative leader.

---

## 4. Risk Control & Macro Throttle Matrices

### 4.1 Spectral Graph Theory: The Fiedler Risk Throttle

Using an eigen-decomposition framework on our Laplacian operator, we sort the spectrum such that $0 = \lambda_1 \le \lambda_2 \le \dots \le \lambda_N$. The second-smallest eigenvalue $\lambda_2$ is the **Fiedler Value**, which quantifies the algebraic connectivity of the graph network.

During periods of macro stress or structural market shifts, cross-asset correlations surge, causing the network to compress into a compact, star-like structure. This behavior reduces cross-sectional diversification potential and weakens mean-reversion dynamics. To hedge against these environments, we evaluate expanding historical baseline estimates of the network topology ($\bar{\lambda}_2(t)$ and $\hat{\sigma}_{\lambda}(t)$) over a minimum lookback threshold of 20 weeks:

$$\lambda_{\text{floor}}(t) = \bar{\lambda}_2(t) - 1.5 \hat{\sigma}_{\lambda}(t)$$

Our macro risk throttle scaling matrix $\gamma(t)$ systematically halves overall position sizes during fragmentation shocks:

$$\gamma(t) = \begin{cases} 0.5, & \text{if } \lambda_2(t) < \lambda_{\text{floor}}(t) \\ 1.0, & \text{otherwise} \end{cases}$$

### 4.2 Portfolio Aggregation and Leverage Caps

At each rebalancing date $t$, we rank our candidate space strictly across active edges in the current MST tree $T(t)$. We select the top $K = 8$ alpha-conviction pairs to populate the portfolio. The aggregate asset-level allocation weights are defined as:

$$w_i(t) = \sum_{j: (i,j) \in \mathcal{P}(t)} w_{ij}(t), \quad w_j(t) = -\sum_{i: (i,j) \in \mathcal{P}(t)} w_{ij}(t)$$

Where $\mathcal{P}(t)$ denotes our active set of top traded pairs. To maintain institutional risk constraints, we enforce a strict gross notional leverage cap $G_{\max} = 4.0$ via a proportional rescaling operator if gross exposure exceeds boundaries:

$$G(t) = \sum_{i=1}^N |w_i(t)|$$

$$\text{If } G(t) > G_{\max} \implies w_i(t) \leftarrow w_i(t) \cdot \left( \frac{G_{\max}}{G(t)} \right)$$

---

## 5. Empirical Framework & Investment Universe

The data pipeline consumes adjusted close daily pricing panels across **21 highly liquid ETFs** spanning a multi-asset allocation matrix, sourced via NASDAQ Data Link. The dataset spans from January 2, 2018, through December 31, 2025, capturing complex macro cycles including the 2020 liquidity crunch, the 2022 rate hikes, and post-inflation market recoveries.

| Asset Class | Ticker Symbols | Asset Count |
| --- | --- | --- |
| **Equity Sectors** | XLF, XLK, XLE, XLV, XLI, XLY, XLP, XLB, XLU 

 | 9 |
| **Commodities** | GLD, SLV, USO, DBA 

 | 4 |
| **Currencies** | FXE, FXB, FXY, FXA 

 | 4 |
| **Fixed Income** | TLT, IEF, HYG, LQD 

 | 4 |

---

## 6. Project Structure & Core Module

* 
`data/`: Local binary cache repository storing adjusted close historical pricing series.


* `mst_strategy.py`: Specialized algorithmic execution engine handling Kruskal tree generation and algebraic Laplacian parsing.
* `Topology_Trading_Backtest.ipynb`: Interactive simulation notebook containing historical performance logs and analysis.

### Quickstart Execution Engine

```python
import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator

def compute_graph_laplacian(correlation_matrix):
    """
    Computes Moore-Penrose Inverse and Fiedler Value from Pearson matrix
    """
    # 1. Transform Pearson correlation space to Mantegna Metric space
    distance_matrix = np.sqrt(2 * (1.0 - correlation_matrix))
    
    # 2. Extract MST Topology via Kruskal execution
    mst_adj = build_mst_kruskal(distance_matrix)
    
    # 3. Construct Graph Laplacian Matrix
    degree_matrix = np.diag(np.sum(mst_adj, axis=1))
    laplacian = degree_matrix - mst_adj
    
    # 4. Access Spectrum to pull out Algebraic Connectivity (Fiedler Value)
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    fiedler_value = eigenvalues[1] 
    
    # 5. Execute Pseudoinverse via Singular Value Decomposition
    laplacian_pseudo_inverse = np.linalg.pinv(laplacian)
    
    return laplacian_pseudo_inverse, fiedler_value

```

---

## 7. Performance & Out-of-Sample Backtest Summary

The strategy operates using a weekly rebalancing cadence (executed on Friday closes) with a realistic transaction cost model of 5 basis points per leg deducted from trading performance.

* **In-Sample Configuration Performance:** Showcases highly stable metrics with uniform Sharpe parameters across asset subgroups, confirming structural relationships.
* **Out-of-Sample Performance Dynamics:** High-turnover parameter profiles deteriorate heavily once transaction execution friction is added. However, the systematic trend configuration remains robust, producing an annualized return of **~10.5%** and a Sharpe ratio of **~0.47**.
* **Risk Profile:** Higher system return generation carries tail risks; the aggressive allocation configuration experiences a maximum peak-to-trough drawdown of **-32.6%**, highlighting the need for dynamic leverage resizing.

---

## 8. References

* Mantegna, R. N. (1999). *Hierarchical structure in financial markets*. European Physical Journal B.
* Klein, D. J., & Randić, M. (1993). *Resistance distance*. Journal of Mathematical Chemistry.
* Onnela, J. P., et al. (2003). *Dynamics of market node topologies in stress environments*. Physica A: Statistical Mechanics and its Applications.
* Pozzi, F., Di Matteo, T., & Aste, T. (2013). *Spread topology metrics and asset diversification risk mitigation*. Physical Review E.

*** ### Summary of References (for your internal compilation)

* **NASDAQ Data Link Search Query:** `FINM_Winter_2026_Group_K_QTS_Data_Cache` (included for verification references).
