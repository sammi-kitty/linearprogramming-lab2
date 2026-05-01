import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as pyplot

ITERATIONS = 100

def main():
    original_data = np.loadtxt(r"SP500.txt")

    # Get average of the historical returns
    historical_avg = np.average(original_data, axis=0)

    # Get variance for the different returns
    variance = np.average(np.square(original_data - historical_avg), axis=0)

    # Get inflation (basic implementation)
    inflation = np.average(historical_avg)

    # Set init guess
    init_guess = np.ones(variance.size) / variance.size

    # Run CGM
    history_agnostic = cgm_implementation(init_guess, historical_avg, variance, inflation, 1)
    history_greedy = cgm_implementation(init_guess, historical_avg, variance, inflation, 2)
    history_nike = cgm_implementation(init_guess, historical_avg, variance, inflation, 3)

    pyplot.plot(history_agnostic, label="Agnostic")
    pyplot.plot(history_greedy, label="Greedy")
    pyplot.plot(history_nike, label="Nike Agnostic")
    pyplot.xlabel('Iteration')
    pyplot.ylabel('Risk')
    pyplot.legend()
    pyplot.show()

def solve_lp(historical_avg, inflation_rate, c):

    # Convert historical_avg @ x >= inflation rate to 
    # -historical_avg @ x <= -inflation_rate
    A_ub = [-historical_avg]
    b_ub = [-inflation_rate]

    # Make sure that sum(x) == 1
    A_eq = [np.ones(25)]
    b_eq = [1]

    bounds = [(0, None)] * 25

    return linprog(
        c = c, A_ub = A_ub, b_ub = b_ub, A_eq = A_eq,
        b_eq = b_eq, bounds = bounds, method = 'highs'
        ).x


def cgm_implementation(init_guess, historical_avg, variance, inflation, stepsize_type):
    history = []
    x = init_guess
    for i in range(1, ITERATIONS + 1):
        c = 2 * variance * x # LP coefficient

        y = solve_lp(historical_avg, inflation, c)

        history.append(
            np.sum(np.square(y) * variance)
        )

        if stepsize_type == 1:
            h = agnostic_step_size(i)
        elif stepsize_type == 2:
            h = greedy_step_size(x, y, variance)
        elif stepsize_type == 3:
            h = nike_agnostic(i)
        else:
            return

        # print("Old Guess:" + str(guess))
        # print(h)

        x = ((1 - h) * x) + (h * y)
    
        # print("New Guess:" + str(guess))
        # print("=" * 30)
    return history


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