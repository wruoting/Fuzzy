import numpy as np
import matplotlib.pyplot as plt
import autograd.numpy as auto_np
from autograd import grad


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


def create_three_point_file():
    f = open("Data/ThreePointPeak/normalized_peak.txt", "w+")
    x = [0, 5, 10]
    y = [1, 5, 1]

    for value in x:
        f.write(str(value))
        f.write(" ")
    f.write(",")
    for value in y:
        f.write(str(value))
        f.write(" ")
    f.close()


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


def create_normalized_file_sample_size_10():
    f = open("Data/NormalizedPeakCenterLowSampleSize/normalized_peak.txt", "w+")
    mu = 0.5
    sigma = 0.2

    x = np.random.normal(loc=mu, scale=sigma, size=60)
    y = np.array(gaussian_distribution(x, mu, sigma))

    for value in x:
        f.write(str(value))
        f.write(" ")
    f.write(",")
    for value in y:
        f.write(str(value))
        f.write(" ")
    f.close()


def create_left_peak_gumbel_file_sample_size_60():
    f = open("Data/LeftPeakCenterLowSampleSize/normalized_peak.txt", "w+")
    mu = 0.5
    sigma = 0.2

    x = np.random.gumbel(loc=1, scale=1, size=60)
    y = np.array(gaussian_distribution(x, mu, sigma))
    for value in x:
        f.write(str(value))
        f.write(" ")
    f.write(",")
    for value in y:
        f.write(str(value))
        f.write(" ")
    f.close()


def create_left_peak_gumbel_file_sample_size_60_higher_sig():
    f = open("Data/LeftPeakCenterHigherSigLowSampleSize/Trim_ABC/normalized_peak.txt", "w+")
    mu = 0.5
    sigma = 0.7

    x = np.random.gumbel(loc=1, scale=1, size=60)
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


def create_right_peak_gumbel_file():
    f = open("Data/RightPeakCenter/normalized_peak.txt", "w+")
    mu = 0.5
    sigma = 1
    x = 1 - np.random.gumbel(loc=1, scale=1, size=400)
    y = np.array(gaussian_distribution(x, mu, sigma))
    for value in x:
        f.write(str(value))
        f.write(" ")
    f.write(",")
    for value in y:
        f.write(str(value))
        f.write(" ")
    f.close()


def create_bimodal_peak_gumbel_file():
    f = open("Data/BimodalPeak/normalized_peak.txt", "w+")
    mu_1 = 0.5
    mu_2 = 5.5
    sigma = 1
    x_1 = np.random.normal(loc=mu_1, scale=sigma, size=200)
    x_2 = np.random.normal(loc=mu_2, scale=sigma, size=200)
    y_1 = np.array(gaussian_distribution(x_1, mu_1, sigma))
    y_2 = np.array(gaussian_distribution(x_2, mu_2, sigma))
    x = np.append(x_1, x_2)
    y = np.append(y_1, y_2)
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
    text_file = open(path, "r")
    lines = text_file.read().split(',')

    x_values = np.array(lines[0].split(' ')[:-1]).astype('float64')
    y_values = np.array(lines[1].split(' ')[:-1]).astype('float64')

    return x_values, y_values
