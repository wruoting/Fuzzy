import numpy as np
import matplotlib.pyplot as plt


def gaussian_distribution(x, mu, sigma):
    variance = np.power(sigma, 2)
    first_term = np.divide(1, np.sqrt(np.multiply(variance, 2*np.pi)))
    final_y = []
    for term in x:
        second_term = np.exp(
                        np.divide(
                            -np.power((term-mu), 2),
                            2*variance
                        ))

        gaussian_y = np.multiply(first_term, second_term)
        randomized_y = np.random.rand(1)[0]
        final_y.append(gaussian_y - randomized_y)
    return final_y


def create_normalized_file():
    f = open("Data/NormalizedPeakCenter/normalized_peak.txt", "w+")
    mu = 0.5
    sigma = 1

    x = np.random.normal(loc=mu, scale=sigma, size=400)
    y = np.array(gaussian_distribution(x, mu, sigma))

    for value in x:
        f.write(str(value))
        f.write(" ")
    f.write(",")
    for value in y:
        f.write(str(value))
        f.write(" ")
    f.close()


def create_left_peak_gumbel_file():
    f = open("Data/LeftPeakCenter/normalized_peak.txt", "w+")
    mu = 0.5
    sigma = 1
    x = np.random.gumbel(loc=1, scale=1, size=400)
    y = np.array(gaussian_distribution(x, mu, sigma))
    for value in x:
        f.write(str(value))
        f.write(" ")
    f.write(",")
    for value in y:
        f.write(str(value))
        f.write(" ")
    f.close()


def create_file(path=None, x_data=None, y_data=None):
    if not path:
        raise ValueError("You must pass a value into the path")
    f = open(path, "w+")
    for value in x_data:
        f.write(str(value))
        f.write(" ")
    f.write(",")
    for value in y_data:
        f.write(str(value))
        f.write(" ")
    f.close()


def open_data(path=None):
    if path:
        text_file = open(path, "r")
    else:
        text_file = open("Data/NormalizedPeakCenter/normalized_peak.txt", "r")
    lines = text_file.read().split(',')
    x_values = np.array(lines[0].split(' ')[:-1]).astype(float)
    y_values = np.array(lines[1].split(' ')[:-1]).astype(float)

    return x_values, y_values


create_left_peak_gumbel_file()


