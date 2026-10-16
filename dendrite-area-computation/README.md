# Postprocessor Tools (Apps)

## How to calculate the irregular dendrites area fraction using the continuous integral method?

[![continuummodelnt2d](https://img.shields.io/badge/postprocessarea1-streamlit-green)](https://continuous-2d-area1.streamlit.app/ )  


# Mathematical Formulation for Continuous Dendrite Area Computation

## Introduction
In phase-field modeling of dendrite growth, the interface between the solid dendrite and the liquid electrolyte is diffuse rather than sharp. Consequently, the "surface area" or cross-sectional area of the dendrite is computed mathematically as a continuous integral over the domain of interest. This is achieved by utilizing a smooth indicator function evaluated via finite element Gauss quadrature.

---

## 1. Phase-Field Variable Normalization
The simulation tracks a phase-field variable, $\eta$, which transitions smoothly from the electrolyte phase ($\eta \approx 0$) to the dendrite phase ($\eta \approx 1$). To standardize this variable across the domain, $\eta$ is normalized into a bounded variable $q \in [0, 1]$:

$$
q = \text{clip}\left( \frac{\eta - \eta_{\text{low}}}{\eta_{\text{high}} - \eta_{\text{low}}}, \, 0.01, \, 0.99 \right)
$$

where $\eta_{\text{low}} = 0.0$ and $\eta_{\text{high}} = 1.0$. The clipping bounds $(0.01, 0.99)$ are applied to prevent numerical singularities at the exact phase limits.

---

## 2. Smooth Indicator Function
Instead of using a discontinuous Heaviside step function, a 5th-order polynomial smooth step function $h(q)$ is used to represent the local volume/area fraction of the dendrite phase:

$$
h(q) = q^3 (6q^2 - 15q + 10) = 6q^5 - 15q^4 + 10q^3
$$

This specific polynomial is chosen because its first derivative is proportional to the standard double-well potential $W(q) = q^2(1-q)^2$:

$$
h'(q) = 30q^2(1-q)^2
$$

Because $h'(0) = 0$ and $h'(1) = 0$, the function maintains $C^1$ continuity (smoothness) across the diffuse interface, ensuring thermodynamic consistency and preventing artificial numerical oscillations during integration.

---

## 3. Continuous Area Formulation
The continuous dendrite area $A_d$ is defined as the integral of the smooth indicator function $h(q)$ over the spatial domain of interest $\Omega$:

$$
A_d = \iint_{\Omega} h(q(x,y)) \, dA
$$

The total area of the region $\Omega$ is simply:

$$
A_{\text{total}} = \iint_{\Omega} 1 \, dA
$$

The **Dendrite Area Fraction** is then calculated as a percentage:

$$
\text{Area Fraction (\%)} = \frac{A_d}{A_{\text{total}}} \times 100
$$

### Spatial Restriction
To measure only the dendrite that has grown *past* the initial electrode boundary into the electrolyte, the integration domain $\Omega$ is restricted to the spatial region where the $x$-coordinate exceeds a threshold $x_{\text{end}}$ (e.g., $x_{\text{end}} = 5.0$):

$$
\Omega = \{ (x,y) \mid x > x_{\text{end}} \}
$$

In the computational implementation, this is enforced at the numerical integration points using an indicator function $\mathbb{I}(x > x_{\text{end}})$.

---

## 4. Finite Element Discretization and Gauss Quadrature
Because the mesh consists of 2D QUAD4 (4-node quadrilateral) elements, the continuous integrals cannot be solved analytically. They are evaluated numerically using isoparametric mapping and $3 \times 3$ Gauss quadrature.

### 4.1 Isoparametric Mapping and Shape Functions
For each element, the physical coordinates $(x,y)$ and the phase-field variable $q$ are mapped from parametric coordinates $(r,s) \in [-1, 1]^2$ using bilinear shape functions $\mathbf{N}(r,s)$:

$$
\mathbf{N}(r,s) = \frac{1}{4} \begin{bmatrix} 
(1-r)(1-s) \\ 
(1+r)(1-s) \\ 
(1+r)(1+s) \\ 
(1-r)(1+s) 
\end{bmatrix}
$$

The physical coordinates and variables at any point $(r,s)$ within the element are interpolated from the nodal values:

$$
x(r,s) = \mathbf{x}^T \mathbf{N}, \quad y(r,s) = \mathbf{y}^T \mathbf{N}
$$
$$
q(r,s) = \mathbf{q}^T \mathbf{N}
$$

where $\mathbf{x}, \mathbf{y}, \mathbf{q}$ are vectors containing the nodal coordinates and nodal phase-field values, respectively.

### 4.2 Jacobian and Differential Area
The differential physical area $dA$ is transformed to the parametric space using the determinant of the Jacobian matrix $\mathbf{J}$:

$$
\mathbf{J} = \begin{bmatrix} 
\frac{\partial x}{\partial r} & \frac{\partial y}{\partial r} \\ 
\frac{\partial x}{\partial s} & \frac{\partial y}{\partial s} 
\end{bmatrix}
$$

The determinant of the Jacobian is computed as:

$$
|\mathbf{J}| = \left| \frac{\partial x}{\partial r}\frac{\partial y}{\partial s} - \frac{\partial x}{\partial s}\frac{\partial y}{\partial r} \right|
$$

Thus, the differential area becomes:

$$
dA = |\mathbf{J}| \, dr \, ds
$$

### 4.3 Numerical Integration
The integral is approximated by summing the evaluated functions at 9 Gauss points $(r_i, s_j)$ with corresponding Legendre weights $w_i$ and $w_j$:

$$
A_d \approx \sum_{e=1}^{N_{\text{elem}}} \sum_{i=1}^{3} \sum_{j=1}^{3} w_i w_j \, h(q(r_i, s_j)) \, |\mathbf{J}(r_i, s_j)| \cdot \mathbb{I}(x(r_i, s_j) > x_{\text{end}})
$$

where $N_{\text{elem}}$ is the total number of QUAD4 elements in the mesh, and $\mathbb{I}$ is the spatial restriction indicator function. The total area $A_{\text{total}}$ is computed identically, but without the $h(q)$ term.
