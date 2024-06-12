import numpy as np

# https://code.visualstudio.com/docs/python/python-tutorial#_run-hello-world


def getFunction(num_samples, m, noise):
    x = np.arange(0, num_samples, dtype=float)
    x[0] = linearFunction(m, [0, 0], noise[0])
    x[1] = linearFunction(m, [x[0], 0], noise[1])
    for i in range(2, num_samples):
        x[i] = linearFunction(m, [x[i - 1], x[i - 2]], noise[i])
    return x


def linearFunction(m, x, noise):
    return sum(m[k]*x[k] for k in range(len(m))) + noise


def generateObservations(n, noise, m):
    if n == 0:
        return noise[0]

    N = n + 1
    x = np.zeros(N)
    x[0] = noise[0]
    x[1] = linearFunction(m, [x[0], 0], noise[1])

    if n == 1:
        print("x[{0}] = {1}*{2} + {3}*{4} + {5} = {6}".format(n,
              m[0], x[n - 1], m[1], 0, noise[n], x[n]))
        return x[n]

    for i in range(2, N):
        x[i] = linearFunction(m, [x[i - 1], x[i - 2]], noise[i])
        print("x[{0}] = {1}*{2} + {3}*{4} + {5} = {6}".format(i,
              m[0], x[i - 1], m[1], x[i - 2], noise[i], x[i]))

    return x[n]


def generateObservations2(n, X):
    if n < 0:
        return 0
    return X[n]


mean = 0
std = 1
num_samples = 2000
noise = np.random.normal(mean, std, size=num_samples)

m1 = 0.6530
m2 = -0.7001
print("noise = {0} | M = {1}".format(noise[:10], [m1, m2]))
print("----------------------")

# x = np.zeros(100)
# print(x)
# x = generateObservations(0, noise, [m1, m2])
# print("x[{0}] = {1}".format(0, x))
# print("----------------------")
# x = generateObservations(1, noise, [m1, m2])
# print("x[{0}] = {1}".format(1, x))
# print("----------------------")
# x = generateObservations(2, noise, [m1, m2])
# print("x[{0}] = {1}".format(2, x))
# print("----------------------")
# x = generateObservations(3, noise, [m1, m2])
# print("x[{0}] = {1}".format(3, x))
# print("----------------------")
# x = generateObservations(4, noise, [m1, m2])
# print("x[{0}] = {1}".format(4, x))
# print("----------------------")
# x = generateObservations(5, noise, [m1, m2])
# print("x[{0}] = {1}".format(5, x))
# print("----------------------")
# x = generateObservations(6, noise, [m1, m2])
# print("x[{0}] = {1}".format(6, x))
