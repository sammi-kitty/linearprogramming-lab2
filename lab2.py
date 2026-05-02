import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as pyplot

ITERATIONS = 1000

def main():
    original_data = np.loadtxt(r"SP500.txt")

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

    # pyplot.figure()

    # pyplot.subplot(121)
    pyplot.plot(history_agnostic[0], label="Agnostic")
    pyplot.plot(history_greedy[0], label="Greedy")
    pyplot.xlabel('Iteration')
    pyplot.ylabel('Risk')
    pyplot.legend()
    pyplot.show()

    '''pyplot.subplot(122)
    # pyplot.plot(history_agnostic[1], label = "Stock portion")
    x_composition_history = history_agnostic[1]
    for i in range(25):
        stock = x_composition_history[:, i]
        pyplot.plot(stock, label=f"Stock x_{i}")
        
    pyplot.xlabel('Iteration')
    pyplot.ylabel('')
    pyplot.legend()
    pyplot.show()'''

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
    return_history = []
    x_composition_history = np.zeros([ITERATIONS, 25])
    x = init_guess
    for i in range(1, ITERATIONS + 1):
        c = 2 * variance * x # LP coefficient

        y = solve_lp(historical_avg, inflation, c)

        return_history.append(
            np.sum(np.square(x) * variance)
        )
        np.insert(x_composition_history, i, x, axis=0)


        if stepsize_type == 1:
            h = agnostic_step_size(i)
        elif stepsize_type == 2:
            h = greedy_step_size(x, y, variance)
        else:
            return


        x = ((1 - h) * x) + (h * y)
 
    # print(x_composition_history)
    return [return_history, x_composition_history, x]

def initial_guess(historical_avg, inflation):
    c = np.ones(25)
    
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