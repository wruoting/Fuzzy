
# program = f
# v = f(p)
# Initialize non parameter real variables:
p = 9  # parameter of this program
v = 3  # input variable 1

Dv_Dp = 0
v = p ** 2 + v*p + v
# Yields: p^2 + 3*p + 3
v = v - 2
# Yields: p^2 + 3*p + 1

if p > v: # True
    v = v + 1

# Yields: p^2 + 3*p + 2

print(v)  # output
