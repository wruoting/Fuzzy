
# program = f
# z = f(p, u)
# dz/dp = df/dp * dp/dp + df/du * du/dp
# Initialize non parameter real variables:
Du_Dp = 0

p = 9  # parameter of this program
u = 3 * p  # input variable 1
z = u**2  # output

# Insert into the program this assignment:
# D_z/D_p = f_1 + f_2 * D_u/D_p
# f_1 = df_dp
# f_2 = df_du
print(z)  # output
