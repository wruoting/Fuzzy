from autograd import grad


def objective(x):
    y = x**3 + 4 * x**2 + 3
    return y


grad_objective = grad(objective)

print(grad_objective(4.0))
