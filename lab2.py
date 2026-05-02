import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as pyplot

ITERATIONS = 1000

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
    inflation = np.average(historical_avg)

    # Set init guess
    init_guess = initial_guess(historical_avg, inflation)

    # Run CGM
    history_agnostic = cgm_implementation(init_guess, historical_avg, variance, inflation, 1)
    history_greedy = cgm_implementation(init_guess, historical_avg, variance, inflation, 2)

    
    # Plot current risk
    pyplot.plot(history_agnostic[0], label="Agnostic")
    pyplot.plot(history_greedy[0], label="Greedy")
    pyplot.xlabel('Iteration')
    pyplot.ylabel('Risk')
    pyplot.title("Risk för $x^{(k)}$ given av CGM, för steglängd $greedy$ och $agnostic$.")
    pyplot.legend()
    pyplot.savefig(f"{ITERATIONS}-iter-risk.png", dpi=400)
    pyplot.clf()

    # Plot current profit
    pyplot.plot(history_agnostic[2], label="Agnostic")
    pyplot.plot(history_greedy[2], label="Greedy")
    pyplot.xlabel('Iteration')
    pyplot.ylabel('Avkastning')
    pyplot.title("Avkastning för $x^{(k)}$ given av CGM, för steglängd $greedy$ och $agnostic$.")
    pyplot.legend()
    pyplot.savefig(f"{ITERATIONS}-iter-profit.png", dpi=400)
    pyplot.clf()

    fig, ax = pyplot.subplots(figsize = (7, 2.5))
    x_composition_history = history_greedy[1]
    i = 0
    for stock in x_composition_history:
        ax.plot(stock, label=f'{name_data[i]}')
        i = i + 1
    ax.set_ylabel('Del av portfölj, andel av 1')
    ax.set_xlabel('Iteration')
    fig.legend(fontsize = 'xx-small', loc='outside right upper', labelspacing=0.2,)
    fig.savefig(f"{ITERATIONS}-iter-composition.png", dpi=400)
    

    agnostic_solution = history_agnostic[3]
    greedy_solution = history_greedy[3]

    # print(f"Agnostic solution: {history_agnostic[3]}")
    # print(f"Greedy solution: {history_greedy[3]}")
    for i in range(25):
        print(
            f'''{name_data[i]} &
            {100 * agnostic_solution[i]:.6f}\% &
            {100 * greedy_solution[i]:.6f}\%
            \\\\ \hline''')
    

def solve_lp(historical_avg, inflation, c):

    # Convert historical_avg @ x >= inflation to 
    # -historical_avg @ x <= -inflation
    A_ub = [-historical_avg]
    b_ub = [-inflation]

    # Make sure that sum(x) == 1
    A_eq = [np.ones(25)]
    b_eq = [1]

    bounds = [(0, None)] * 25

    return linprog(
        c = c, A_ub = A_ub, b_ub = b_ub, A_eq = A_eq,
        b_eq = b_eq, bounds = bounds, method = 'highs'
        ).x


def cgm_implementation(init_guess, historical_avg, variance, inflation, stepsize_type):
    risk_history = []
    x_composition_history = np.empty([ITERATIONS, 25])
    profit_history = []
    x = init_guess
    has_converged = False
    for i in range(1, ITERATIONS + 1):
        c = 2 * variance * x # LP coefficient

        y = solve_lp(historical_avg, inflation, c)

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
        else:
            return

        
        if np.linalg.norm(x_composition_history[i-1] - x_composition_history[i-2]) == 0 and not(has_converged) and i > 5:
            print(f"Converged at iteration {i} for type {stepsize_type}")
            has_converged = True
            
            


        x = ((1 - h) * x) + (h * y)

    composition_history_star = [None] * 25
    for i in range(25):
        composition_history_star[i] = x_composition_history[:, i]
    
    return [risk_history, composition_history_star, profit_history, x]

def initial_guess(historical_avg, inflation):
    '''
    c = np.ones(25) + 1
    
    # Convert historical_avg @ x >= inflation to 
    # -historical_avg @ x <= -inflation
    A_ub = [-historical_avg]
    b_ub = [-inflation]

    # Make sure that sum(x) == 1
    A_eq = [np.ones(25)]
    b_eq = [1]

    bounds = [(0, None)] * 25

    return linprog(
        c = c, A_ub = A_ub, b_ub = b_ub, A_eq = A_eq,
        b_eq = b_eq, bounds = bounds, method = 'highs'
        ).x
    '''
    naive_start = np.ones(25) / 25
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

def nike_agnostic(i):
    return 2 / (i**1.5 + 2)


# Pseudo code:
# - select initial guess
# - 

main()