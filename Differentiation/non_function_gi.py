
# program = f
# v = f(p)
# Initialize non parameter real variables:
def f(p):
    # p = 9.01  # parameter of this program
    v_1 = 3  # input variable 1
    v_2 = 4
    v_1_i = 3  # input variable 1
    v_2_i = 4

    v_1 = p ** 2 + v_1*p + v_1*v_2
    # f' functions
    Df_Dp = 2*p+v_1
    Df_Dv2 = v_1
    Df_Dv1 = p + v_2

    # dv_dp functions
    Dv1_Dp = 2*p+v_1 # same as df_dp
    Dv2_Dp = 0  # v_2 has not changed during this time

    # Yields: p^2 + 3*p + 12
    v_1 = v_1 - 2

    # f' functions
    Df_Dp = 2*p+v_1
    Df_Dv2 = v_1
    Df_Dv1 = p + v_2

    # dv_dp functions
    Dv1_Dp = 2*p+v_1 # same as df_dp
    Dv2_Dp = 0  # v_2 has not changed during this time


    # Yields: p^2 + 3*p + 10
    v_1 = v_1 + v_2

    # f' ofunctions
    Df_Dp = 2*p+v_1_i
    Df_Dv2 = v_1_i+1
    Df_Dv1 = p + v_2_i

    # dv_dp functions
    Dv1_Dp = 2*p+v_1_i # same as df_dp
    Dv2_Dp = 0  # v_2 has not changed during this time


    # Yields: p^2 + 3*p + 14
    print(Df_Dp)
    print(Df_Dv1)
    print(Dv1_Dp)
    final_equation_differentiation = Df_Dp + Df_Dv1 * Dv1_Dp + Df_Dv2 * Dv2_Dp
    print(final_equation_differentiation)
    print(v_1)
    return v_1

f(9)
# f(9.01)
