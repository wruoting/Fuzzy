# Create a polynomial and differentiate
import inspect
import argparse

def paramaterize(func):

    all_parameters = {}
    all_parameters.update({'function': func.__name__})
    parameters = {}

    def func_wrapper(*args):
        output, inner_vars = func(*args)
        sig = inspect.signature(func)
        for key in sig.parameters:
            parameters[key] = inner_vars[key]
            del inner_vars[key]
        all_parameters.update({'parameters': parameters})
        all_parameters.update({'output': output})
        all_parameters.update({'local_vars': inner_vars})
        return all_parameters
    return func_wrapper


@paramaterize
def polynomial(x):
    v_1 = x**3
    v_2 = 2*(x**2)
    v_3 = 5*x
    v_out = v_1 + v_2 + v_3
    return v_out, locals()


print(polynomial(2))
