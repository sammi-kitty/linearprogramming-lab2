import numpy as np
from scipy.optimize import linprog

# ========== 1. LOAD AND PROCESS DATA ==========
data = np.loadtxt(r'/home/sammiqueen/Code/linjärprog-python/lab1-python/SP500.txt')

print("Shape: ", data.shape)
T, N = data.shape  # T = number of months, N = number of companies

# Calculate column statistics
column_sums = data.sum(axis=0)
column_averages = column_sums / T  # µ_n (expected returns)
column_variances = np.var(data, axis=0)  # σ²_n (risk)

print(f"\nNumber of companies (N): {N}")
print(f"Number of months (T): {T}")
print(f"Expected returns (µ): {column_averages[:5]}...")  # Show first 5
print(f"Variances (σ²): {column_variances[:5]}...")

# ========== 2. SET PARAMETERS ==========
# First experiment: α = average return
alpha_1 = np.mean(column_averages)  # µ_avg
print(f"\nα (minimum expected return): {alpha_1:.6f}")

# Second experiment: more ambitious goal
mu_max = np.max(column_averages)
alpha_2 = 0.5 * (alpha_1 + mu_max)
print(f"α ambitious: {alpha_2:.6f}")

# ========== 3. INITIAL FEASIBLE SOLUTION ==========
# Equal allocation (sums to 1, all non-negative)
x = np.ones(N) / N
print(f"\nInitial portfolio: {x[:5]}...")
print(f"Check sum: {np.sum(x):.4f}")
print(f"Initial expected return: {np.sum(column_averages * x):.6f}")
print(f"Initial risk: {np.sum(column_variances * x**2):.6f}")

# ========== 4. CGM IMPLEMENTATION ==========
def solve_auxiliary_lp(c, mu, alpha):
    """
    Solve the auxiliary linear program:
    minimize c^T x
    subject to: sum(mu_n * x_n) >= alpha
                sum(x_n) = 1
                x_n >= 0
    """
    N = len(mu)
    
    # Constraint: sum(mu_n * x_n) >= alpha  →  -sum(mu_n * x_n) <= -alpha
    A_ub = [-mu]  # Negative because we need >= constraint
    b_ub = [-alpha]
    
    # Equality constraint: sum(x_n) = 1
    A_eq = [np.ones(N)]
    b_eq = [1]
    
    # Bounds: x_n >= 0
    bounds = [(0, None)] * N
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                     bounds=bounds, method='highs')
    
    if not result.success:
        print(f"Warning: LP solver failed - {result.message}")
        return None
    
    return result.x

def greedy_step_size(x, y, sigma2):
    """
    Find optimal η ∈ [0,1] that minimizes f((1-η)x + ηy)
    where f(x) = sum(sigma2_n * x_n^2)
    """
    # f(η) = sum(sigma2_n * ((1-η)x_n + η y_n)^2)
    # Expand quadratic in η: a*η^2 + b*η + c
    diff = y - x
    a = np.sum(sigma2 * diff**2)
    b = 2 * np.sum(sigma2 * x * diff)
    # c = sum(sigma2 * x^2) - not needed for minimization
    
    if a <= 0:  # Linear or concave case
        # Check endpoints
        f0 = np.sum(sigma2 * x**2)
        f1 = np.sum(sigma2 * y**2)
        return 0 if f0 < f1 else 1
    
    # Unconstrained minimum: η* = -b/(2a)
    eta_star = -b / (2 * a)
    
    # Clamp to [0,1]
    return np.clip(eta_star, 0, 1)

# Run CGM for different settings
def run_cgm(mu, sigma2, alpha, max_iter=100, step_type='agnostic'):
    """
    Run Conditional Gradient Method
    step_type: 'agnostic' or 'greedy'
    """
    N = len(mu)
    x = np.ones(N) / N  # Initial feasible solution
    risk_history = []
    
    print(f"\n--- Running CGM with {step_type} step sizes ---")
    print(f"Target return α = {alpha:.6f}")
    
    for k in range(1, max_iter + 1):
        # Step 1: Linearize objective -> coefficients for LP
        c = 2 * sigma2 * x
        
        # Step 2: Solve auxiliary LP
        y = solve_auxiliary_lp(c, mu, alpha)
        
        if y is None:
            print(f"Failed at iteration {k}")
            break
        
        # Step 3: Choose step size
        if step_type == 'greedy':
            eta = greedy_step_size(x, y, sigma2)
        else:  # agnostic
            eta = 2 / (k + 1)
        
        # Step 4: Update
        x = (1 - eta) * x + eta * y
        
        # Calculate and store current risk
        current_risk = np.sum(sigma2 * x**2)
        risk_history.append(current_risk)
        
        # Print progress every 100 iterations
        if k % 100 == 0 or k == 1:
            expected_return = np.sum(mu * x)
            print(f"Iter {k:4d}: Risk = {current_risk:.8f}, "
                  f"Return = {expected_return:.6f}, η = {eta:.4f}")
    
    return x, risk_history

# ========== 5. RUN EXPERIMENTS ==========

# Experiment 1: α = average return, agnostic step sizes
print("\n" + "="*60)
print("EXPERIMENT 1: α = average return, agnostic step sizes")
print("="*60)
x_final_1a, risk_hist_1a = run_cgm(column_averages, column_variances, 
                                     alpha_1, max_iter=100, step_type='agnostic')

# Experiment 2: α = average return, greedy step sizes
print("\n" + "="*60)
print("EXPERIMENT 2: α = average return, greedy step sizes")
print("="*60)
x_final_1b, risk_hist_1b = run_cgm(column_averages, column_variances, 
                                     alpha_1, max_iter=100, step_type='greedy')

# Experiment 3: α = ambitious, agnostic step sizes
print("\n" + "="*60)
print("EXPERIMENT 3: α = ambitious return, agnostic step sizes")
print("="*60)
x_final_2a, risk_hist_2a = run_cgm(column_averages, column_variances, 
                                     alpha_2, max_iter=100, step_type='agnostic')

# ========== 6. RESULTS AND ANALYSIS ==========

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)

print(f"\nExperiment 1 (α = {alpha_1:.6f}, agnostic):")
print(f"  Final risk (variance): {risk_hist_1a[-1]:.8f}")
print(f"  Final expected return: {np.sum(column_averages * x_final_1a):.6f}")
print(f"  Number of non-zero investments: {np.sum(x_final_1a > 1e-6)}")

print(f"\nExperiment 2 (α = {alpha_1:.6f}, greedy):")
print(f"  Final risk (variance): {risk_hist_1b[-1]:.8f}")
print(f"  Final expected return: {np.sum(column_averages * x_final_1b):.6f}")
print(f"  Number of non-zero investments: {np.sum(x_final_1b > 1e-6)}")

print(f"\nExperiment 3 (α = {alpha_2:.6f}, agnostic):")
print(f"  Final risk (variance): {risk_hist_2a[-1]:.8f}")
print(f"  Final expected return: {np.sum(column_averages * x_final_2a):.6f}")
print(f"  Number of non-zero investments: {np.sum(x_final_2a > 1e-6)}")

# Compare risk levels
print(f"\nRisk comparison:")
print(f"  Higher α ({alpha_2:.6f}) vs lower α ({alpha_1:.6f}):")
risk_ratio = risk_hist_2a[-1] / risk_hist_1a[-1]
print(f"  Risk increased by factor: {risk_ratio:.2f}")
if risk_ratio > 1:
    print("  → Higher expected return requires higher risk (reasonable)")
else:
    print("  → Unexpected result - check your data")

# Display top 5 investments in final portfolio
print("\nTop 5 investments in final portfolio (Experiment 1):")
top_indices = np.argsort(x_final_1a)[-5:][::-1]
for i, idx in enumerate(top_indices):
    print(f"  Company {idx}: {x_final_1a[idx]:.4f} "
          f"(µ={column_averages[idx]:.6f}, σ²={column_variances[idx]:.8f})")

# Optional: Save results
np.savetxt('final_portfolio.csv', x_final_1a, delimiter=',')
print("\nResults saved to 'final_portfolio.csv'")

import matplotlib.pyplot as plt
plt.plot(risk_hist_1a, label='Agnostic α1')
plt.plot(risk_hist_1b, label='Greedy α1')
plt.plot(risk_hist_2a, label='Agnostic α2')
plt.xlabel('Iteration')
plt.ylabel('Risk')
plt.legend()
plt.show()