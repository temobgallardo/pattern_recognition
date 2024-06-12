import pre_processing as pp
import weiner_predictive_filter as wpf
import numpy as np
import matplotlib.pyplot as plt

mean = 0
std = 1
num_samples = 2000
noise = np.random.normal(mean, std, size=num_samples)

MU = 0.001
number_of_coefficients = 2
M = [0.6530, -0.7001]
X = pp.getFunction(num_samples, M, noise)
plt.plot(X, 'g', label='x')
plt.show()

WA = np.zeros(shape=(num_samples, number_of_coefficients))
WA[0, :] = np.array([0.5, -0.5])
# wa = adaptiveWeinerFilter(wa, mu, noise, [0.6530, -0.7001])
[WA, E] = wpf.adaptiveWeinerFilter(WA, MU, X)

plt.plot(WA)
plt.show()

yn, errors = wpf.predictedSignal(X, WA, MU)

# print(wa[num_samples - 1, :])
plt.plot(yn)
plt.plot(X)
plt.show()

plt.plot(errors)
plt.show()


# errors = [wpf.error2(i - 1, wa[i - 1, :], X)
#           for i in range(1, num_samples + 1)]
# print(errors[1:100])
# plt.plot(errors)
# plt.show()
