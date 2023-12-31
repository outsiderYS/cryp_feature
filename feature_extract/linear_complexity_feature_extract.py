import math
import os
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt

# from scipy.special import gamma, gammainc, gammaincc
from gamma_functions import *


def berelekamp_massey(bits):
    n = len(bits)
    b = [0 for x in bits]  # initialize b and c arrays
    c = [0 for x in bits]
    b[0] = 1
    c[0] = 1

    L = 0
    m = -1
    N = 0
    while (N < n):
        # compute discrepancy
        d = bits[N]
        for i in range(1, L + 1):
            d = d ^ (c[i] & bits[N - i])
        if (d != 0):  # If d is not zero, adjust poly
            t = c[:]
            for i in range(0, n - N + m):
                c[N - m + i] = c[N - m + i] ^ b[i]
            if (L <= (N / 2)):
                L = N + 1 - L
                m = N
                b = t
        N = N + 1
    # Return length of generator and the polynomial
    return L, c[0:L]


#该功能需要输入大于10^6，用于4kb输入需调整patternlen大小，如256
def linear_complexity_test(bits, patternlen=None):
    bits = np.unpackbits(np.frombuffer(bits, dtype=np.uint8))
    bits = list(bits)

    n = len(bits)
    # Step 1. Choose the block size
    if patternlen != None:
        M = patternlen
    else:
        if n < 1000000:
            print("Error. Need at least 10^6 bits")
            exit()
        M = 512
    K = 6
    N = int(math.floor(n / M))
    # Step 2 Compute the linear complexity of the blocks
    LC = list()
    for i in range(N):
        x = bits[(i * M):((i + 1) * M)]
        LC.append(berelekamp_massey(x)[0])

    # Step 3 Compute mean
    a = float(M) / 2.0
    b = ((((-1) ** (M + 1)) + 9.0)) / 36.0
    c = ((M / 3.0) + (2.0 / 9.0)) / (2 ** M)
    mu = a + b - c

    T = list()
    for i in range(N):
        x = ((-1.0) ** M) * (LC[i] - mu) + (2.0 / 9.0)
        T.append(x)

    # Step 4 Count the distribution over Ticket
    v = [0, 0, 0, 0, 0, 0, 0]
    for t in T:
        if t <= -2.5:
            v[0] += 1
        elif t <= -1.5:
            v[1] += 1
        elif t <= -0.5:
            v[2] += 1
        elif t <= 0.5:
            v[3] += 1
        elif t <= 1.5:
            v[4] += 1
        elif t <= 2.5:
            v[5] += 1
        else:
            v[6] += 1

    # Step 5 Compute Chi Square Statistic
    pi = [0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]
    chisq = 0.0
    for i in range(K + 1):
        chisq += ((v[i] - (N * pi[i])) ** 2.0) / (N * pi[i])
    # Step 6 Compute P Value
    P = gammaincc((K / 2.0), (chisq / 2.0))
    success = (P >= 0.01)
    return P


def linear_complexity_analysis(dir, block_num, save_path):
    file_count = 0

    for root, dirs, files in os.walk(dir):
        for file in files:
            file_count += 1

    q_values = []

    for i in range(0, block_num):
        file_path = dir + '/chunk_{}.bin'.format(i + 1)
        with open(file_path, "rb") as file:
            content = file.read()
        q_value = linear_complexity_test(content, patternlen=256)
        q_values.append(round(q_value, 2))
    counter = Counter(q_values)
    fig, ax = plt.subplots(figsize=(8, 6))
    x = list(counter.keys())
    y = list(counter.values())
    ax.bar(x, y, width=0.01, edgecolor='black')
    save_path = save_path + '/linear_complexity.png'
    plt.savefig(save_path)


if __name__ == "__main__":
    bits = [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0]
    L, poly = berelekamp_massey(bits)

    file_path = '../dataset/ciphertext/4kb/AES_ECB/total.bin'
    with open(file_path, "rb") as file:
        content = file.read()
    bits = content[0:128000]

    success, p, _ = linear_complexity_test(bits, patternlen=500)

    print("L =", L)
    print("p = ", p)
