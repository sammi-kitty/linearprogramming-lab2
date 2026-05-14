import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as pyplot

ITERATIONS = 1000

N = 25

def main():
    original_data = np.loadtxt(r"SP500.txt")
    name_data = []
    with open("stock_names.txt", "r") as file:
        for line in file:
            name_data.append(str(line))

    # Get average of the historical returns
    historical_avg = np.average(original_data, axis=0)

    # Get variance for the different returns
    variance = np.average(np.square(original_data - historical_avg), axis=0)

    # Get inflation (basic implementation)
    alpha1 = np.average(historical_avg)
    alpha2 = (1/2) * (np.average(historical_avg) + np.max(historical_avg))

    # Set init guess
    init_guess = initial_guess(historical_avg, alpha2)
    print(np.sum(historical_avg * init_guess) - alpha2)

    # Run CGM
    alpha1_history_agnostic = cgm_implementation(init_guess, historical_avg, variance, alpha1, 1)
    alpha1_history_greedy = cgm_implementation(init_guess, historical_avg, variance, alpha1, 2)

    alpha2_history_agnostic = cgm_implementation(init_guess, historical_avg, variance, alpha2, 1)
    alpha2_history_greedy = cgm_implementation(init_guess, historical_avg, variance, alpha2, 2)

    # Plot graphs
    plotting(alpha1_history_greedy, alpha1_history_agnostic, alpha2_history_greedy, alpha2_history_agnostic, name_data)

    # Print LaTeX-ready tabular of solutions
    alpha1_agnostic_solution = alpha1_history_agnostic[3]
    alpha1_greedy_solution = alpha1_history_greedy[3]
    alpha2_agnostic_solution = alpha2_history_agnostic[3]
    alpha2_greedy_solution = alpha2_history_greedy[3]
    for i in range(N):
        print(
            f'''{name_data[i]} &
            {100 * alpha1_agnostic_solution[i]:.6f}\% &
            {100 * alpha1_greedy_solution[i]:.6f}\% &
            {100 * alpha2_agnostic_solution[i]:.6f}\% &
            {100 * alpha2_greedy_solution[i]:.6f}\%
            \\\\ \hline''')

    '''
    print(f"Avkastning för alpha1 agnostic: {np.sum(historical_avg * alpha1_agnostic_solution):.10f}")
    print(f"Avkastning för alpha1 greedy: {np.sum(historical_avg * alpha1_greedy_solution):.10f}")
    print(f"Avkastning för alpha2 agnostic: {np.sum(historical_avg * alpha2_agnostic_solution):.10f}")
    print(f"Avkastning för alpha2 greedy: {np.sum(historical_avg * alpha2_greedy_solution):.10f}")

    print(f"Risk för alpha1 agnostic: {np.sum(variance * np.square(alpha1_agnostic_solution)):.10f}")
    print(f"Risk för alpha1 greedy: {np.sum(variance * np.square(alpha1_greedy_solution)):.10f}")
    print(f"Risk för alpha2 agnostic: {np.sum(variance * np.square(alpha2_agnostic_solution)):.10f}")
    print(f"Risk för alpha2 greedy: {np.sum(variance * np.square(alpha2_greedy_solution)):.10f}")
    '''

def solve_lp(historical_avg, inflation, c):

    # Convert historical_avg @ x >= inflation to 
    # -historical_avg @ x <= -inflation
    A_ub = [-historical_avg]
    b_ub = [-inflation]

    # Make sure that sum(x) == 1
    A_eq = [np.ones(N)]
    b_eq = [1]

    bounds = [(0, None)] * N

    return linprog(
        c = c, A_ub = A_ub, b_ub = b_ub, A_eq = A_eq,
        b_eq = b_eq, bounds = bounds, method = 'highs'
        ).x

def cgm_implementation(init_guess, historical_avg, variance, inflation, stepsize_type):
    risk_history = []
    x_composition_history = np.empty([ITERATIONS, N])
    profit_history = []
    x = init_guess
    has_converged = False
    for i in range(1, ITERATIONS + 1):
        c = 2 * variance * x # LP coefficient

        y = solve_lp(historical_avg, inflation, c)

        # Save values to history (for plotting)
        risk_history.append(
            np.sum(np.square(x) * variance)
        )
        profit_history.append(
            np.sum(x * historical_avg)
        )
        x_composition_history[i - 1] = x

        if stepsize_type == 1:
            h = agnostic_step_size(i)
        elif stepsize_type == 2:
            h = greedy_step_size(x, y, variance)
            # if not(has_converged) and i > 8000:
            #    print(f"{i} & {h:.8f} \\\\ \hline")
        else:
            return

        # Print if algorithm has converged (difference of norm of x_i and x_{i-1} = 0) - only used for debugging purposes
        if np.linalg.norm(x_composition_history[i-1] - x_composition_history[i-2]) == 0 and not(has_converged) and i > 5:
            print(f"Converged at iteration {i} for type {stepsize_type} with alpha {inflation}")
            has_converged = True

        # Update value of x
        x = ((1 - h) * x) + (h * y)

    # Rearrange composition history to be used in plotting
    composition_history_star = [None] * N
    for i in range(N):
        composition_history_star[i] = x_composition_history[:, i]
    
    return [risk_history, composition_history_star, profit_history, x]

def initial_guess(historical_avg, inflation):
    naive_start = np.zeros(25)
    naive_start[18 - 1] = 1
    return naive_start

def greedy_step_size(x, y, variance):
    # BEWARE - AI GENERATED
    """
    Find optimal h ∈ [0,1] that minimizes f((1-h)x + hy)
    where f(x) = sum(variance_n * x_n^2)
    """
    # f(h) = sum(variance_n * ((1-h)x_n + h y_n)^2)
    # Expand quadratic in h: a*h^2 + b*h + c
    diff = y - x
    a = np.sum(variance * diff**2)
    b = 2 * np.sum(variance * x * diff)
    # c = sum(variance * x^2) - not needed for minimization
    
    if a <= 0:  # Linear or concave case
        # Check endpoints
        f0 = np.sum(variance * np.square(x))
        f1 = np.sum(variance * np.square(y))
        return 0 if f0 < f1 else 1
    
    # Unconstrained minimum: h* = -b/(2a)
    h_star = -b / (2 * a)
    
    # Clamp to [0,1]
    return np.clip(h_star, 0, 1)

def agnostic_step_size(i):
    return 2 / (i + 1)

def plotting(alpha1_history_greedy, alpha1_history_agnostic, alpha2_history_greedy, alpha2_history_agnostic, name_data):
    # Plot risk
    pyplot.plot(alpha1_history_agnostic[0], label="Agnostic $\\alpha_1$")
    pyplot.plot(alpha1_history_greedy[0], label="Greedy $\\alpha_1$")
    pyplot.plot(alpha2_history_agnostic[0], label="Agnostic $\\alpha_2$")
    pyplot.plot(alpha2_history_greedy[0], label="Greedy $\\alpha_2$")
    pyplot.xlabel('Iteration $k$')
    pyplot.ylabel('Risk')
    pyplot.title("Risk för $x^{(k)}$ given av CGM, för steglängd $greedy$ och $agnostic$.")
    pyplot.legend()
    pyplot.savefig(f"risk/{ITERATIONS}-iter-risk.png", dpi=400)
    pyplot.clf()

    # Plot profit
    pyplot.plot(alpha1_history_agnostic[2], label="Agnostic $\\alpha_1$")
    pyplot.plot(alpha1_history_greedy[2], label="Greedy $\\alpha_1$")
    pyplot.plot(alpha2_history_agnostic[2], label="Agnostic $\\alpha_2$")
    pyplot.plot(alpha2_history_greedy[2], label="Greedy $\\alpha_2$")
    pyplot.xlabel('Iteration $k$')
    pyplot.ylabel('Avkastning')
    pyplot.title("Avkastning för $x^{(k)}$ given av CGM, för steglängd $greedy$ och $agnostic$.")
    pyplot.legend()
    pyplot.savefig(f"profit/{ITERATIONS}-iter-profit.png", dpi=400)
    pyplot.clf()

    # Plot composition (one for agnostic and one for greedy)
    # Use different linestyles since colours run out
    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    x_composition_history = alpha1_history_greedy[1]
    i = 0
    for stock in x_composition_history:
        if i < 10:
            ax.plot(stock, label=f'$x_{{ {i + 1} }}$, {name_data[i]}')
        elif 10 <= i < 20:
            ax.plot(stock, label=f'$x_{{ {i + 1} }}$, {name_data[i]}', linestyle='--')
        else:
            ax.plot(stock, label=f'$x_{{ {i + 1} }}$, {name_data[i]}', linestyle=':')
        i = i + 1
    ax.set_ylabel('Del av portfölj, andel av 1')
    ax.set_xlabel('Iteration $k$')
    ax.set_title('Portföljkomposition $x^{(k)}$, greedy, $\\alpha_1$')
    fig.legend(fontsize = 'xx-small', loc='outside right upper', labelspacing=0.2,)
    fig.savefig(f"composition/{ITERATIONS}-iter-composition-greedy-alpha1.png", dpi=400)

    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    x_composition_history = alpha1_history_agnostic[1]
    i = 0
    for stock in x_composition_history:
        if i < 10:
            ax.plot(stock, label=f'$x_{{ {i + 1} }}$, {name_data[i]}')
        elif 10 <= i < 20:
            ax.plot(stock, label=f'$x_{{ {i + 1} }}$, {name_data[i]}', linestyle='--')
        else:
            ax.plot(stock, label=f'$x_{{ {i + 1} }}$, {name_data[i]}', linestyle=':')
        i = i + 1
    ax.set_ylabel('Del av portfölj, andel av 1')
    ax.set_xlabel('Iteration $k$')
    ax.set_title('Portföljkomposition $x^{(k)}$, agnostic, $\\alpha_1$')
    fig.legend(fontsize = 'xx-small', loc='outside right upper', labelspacing=0.2,)
    fig.savefig(f"composition/{ITERATIONS}-iter-composition-agnostic-alpha1.png", dpi=400)

    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    x_composition_history = alpha2_history_greedy[1]
    i = 0
    for stock in x_composition_history:
        if i < 10:
            ax.plot(stock, label=f'$x_{{ {i + 1} }}$, {name_data[i]}')
        elif 10 <= i < 20:
            ax.plot(stock, label=f'$x_{{ {i + 1} }}$, {name_data[i]}', linestyle='--')
        else:
            ax.plot(stock, label=f'$x_{{ {i + 1} }}$, {name_data[i]}', linestyle=':')
        i = i + 1
    ax.set_ylabel('Del av portfölj, andel av 1')
    ax.set_xlabel('Iteration $k$')
    ax.set_title('Portföljkomposition $x^{(k)}$, greedy, $\\alpha_2$')
    fig.legend(fontsize = 'xx-small', loc='outside right upper', labelspacing=0.2,)
    fig.savefig(f"composition/{ITERATIONS}-iter-composition-greedy-alpha2.png", dpi=400)

    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    x_composition_history = alpha2_history_agnostic[1]
    i = 0
    for stock in x_composition_history:
        if i < 10:
            ax.plot(stock, label=f'$x_{{ {i + 1} }}$, {name_data[i]}')
        elif 10 <= i < 20:
            ax.plot(stock, label=f'$x_{{ {i + 1} }}$, {name_data[i]}', linestyle='--')
        else:
            ax.plot(stock, label=f'$x_{{ {i + 1} }}$, {name_data[i]}', linestyle=':')
        i = i + 1
    ax.set_ylabel('Del av portfölj, andel av 1')
    ax.set_xlabel('Iteration $k$')
    ax.set_title('Portföljkomposition $x^{(k)}$, agnostic, $\\alpha_2$')
    fig.legend(fontsize = 'xx-small', loc='outside right upper', labelspacing=0.2,)
    fig.savefig(f"composition/{ITERATIONS}-iter-composition-agnostic-alpha2.png", dpi=400)

main()