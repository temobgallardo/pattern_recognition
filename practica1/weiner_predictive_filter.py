import numpy as np
import pre_processing as pp


def lms(n, w, mu, X):
    xn = np.array([pp.generateObservations2(n - 1, X),
                  pp.generateObservations2(n - 2, X)])
    E = error(n, w, X)
    print("W[n] = w[n - 1] - MU*x[n - 1]*E(n - 1) = ")
    W = w + mu*xn*E
    W2 = np.array([0.0, 0.0])
    for k in range(len(w)):
        W2[k] = w[k] + xn[k]*E*mu
    return [W, E]


def error(n, w, X):
    xn = pp.generateObservations2(n, X)
    xn_1 = pp.generateObservations2(n - 1, X)
    xn_2 = pp.generateObservations2(n - 2, X)
    print("error[n] = x[n] - (w[1]*x[n - 1] + w[2]*x[n - 2]) = ")
    estimacion = w[0]*xn_1 + w[1]*xn_2
    print("error[{0}] = {1} - ({2}*{3} + {4}*{5}) = {6}"
          .format(n + 1, xn, w[0], xn_1, w[1], xn_2, xn - estimacion))
    return xn - estimacion


def adaptiveWeinerFilter(wa, mu, X):
    for n in range(1, len(wa)):
        wa[n, :], errors = lms(n - 1, wa[n - 1, :], mu, X)
    return [wa, errors]


def predictedSignal(x, wa, mu):
    wa, errors = adaptiveWeinerFilter(wa, mu, x)
    yn = np.zeros(pp.num_samples)
    W1 = wa[pp.num_samples - 1, 0]
    # print(W1)
    W2 = wa[pp.num_samples - 1, 1]
    # print(W2)
    yn[0] = pp.linearFunction([W1, W2], [0, 0], 0)
    yn[1] = W1*x[0] + 0
    for n in range(2, len(wa)):
        yn[n] = W1*x[n - 1] + W2*x[n - 2]
    return [yn, errors]
