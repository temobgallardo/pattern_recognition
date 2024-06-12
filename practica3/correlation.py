import numpy as np

# Reference: Optimal Wiener filter - Conceptual view, SPS Education, can be found at: https://youtu.be/NOAIrmadDVM?t=206


def crossCorrelationVector(xObservations, numberOfWeights=12, samples=8):
    return [correlation(xObservations, k, samples) for k in range(numberOfWeights)]


def correlation(signal, k, samples=8):
    resultToSum = [multiple(signal, i, k) for i in range(samples)]
    return sum(resultToSum)/samples


def multiple(signal, n, k):
    return signal[n + k]*signal[n]


def autoCorrelationMatrix(xObservations, numberOfWeights=12, samples=8):
    ac = np.zeros([2, 2], dtype=float)
    for row in range(numberOfWeights):
        for col in range(numberOfWeights):
            k = -row + col
            sc = correlation(xObservations, k, samples)
            ac[row][col] = sc
    return ac


def createAutoCorrelationMatrix(acVector, vOrder):
    ac = np.zeros([vOrder, vOrder], dtype=float)
    for i in range(vOrder):
        for j in range(vOrder):
            k = np.abs(i - j)
            ac[i][j] = acVector[k]
    return ac


def optimalWeightVector(xObservations, numWeinerScalars=[], samples=8):
    ac = autoCorrelationMatrix(
        xObservations, numWeinerScalars, samples - len(numWeinerScalars))

    aci = np.linalg.inv(ac)

    ccm = crossCorrelationVector(xObservations, numWeinerScalars,
                                 samples - len(numWeinerScalars))

    ccmMatrix = np.array(ccm).reshape(len(ccm), 1)
    return [multiplyMatrix(aci, ccmMatrix)]


def multiplyMatrix(A, B):
    rowsA = len(A)
    colsA = len(A[0])

    rowsB = len(B)
    colsB = len(B[0])

    if colsA != rowsB:
        return "Lenght mismatch A[{0}, {1}] vs B[{2}, {3}]. Cannot multiply A and B.".format(str(rowsA), str(colsA), str(rowsB), str(colsB))

    C = np.zeros((rowsA, colsB), dtype=float)
    for row in range(rowsA):
        for col in range(colsA):
            for elt in range(colsB):
                C[row, elt] += A[row, col] * B[col, elt]

    return C
