from scipy.optimize import fsolve

def equations(vars):
    x, y = vars
    eq1 = 1 - 400*x*(x**2 + y**2 - 1)
    eq2 = 1 - 400*y*(x**2 + y**2 - 1)
    return [eq1, eq2]

initial_guess = [0, 0]
result = fsolve(equations, initial_guess)

print("Critical Point (x, y):", result)
