import math
import os
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt

# from scipy.special import gamma, gammainc, gammaincc
from gamma_functions import *


def int2patt(n, m):
    pattern = list()
    for i in range(m):
        pattern.append((n >> i) & 1)
    return pattern


def countpattern(patt, bits, n):
    thecount = 0
    for i in range(n):
        match = True
        for j in range(len(patt)):
            if patt[j] != bits[i + j]:
                match = False
        if match:
            thecount += 1
    return thecount


def psi_sq_mv1(m, n, padded_bits):
    counts = [0 for i in range(2 ** m)]
    for i in range(2 ** m):
        pattern = int2patt(i, m)
        count = countpattern(pattern, padded_bits, n)
        counts.append(count)

    psi_sq_m = 0.0
    for count in counts:
        psi_sq_m += (count ** 2)
    psi_sq_m = psi_sq_m * (2 ** m) / n
    psi_sq_m -= n
    return psi_sq_m


# 需要根据输入的数据，控制patternlen,注意输出为两个p-value
def serial_test(bits, patternlen=None):
    bits = np.unpackbits(np.frombuffer(bits, dtype=np.uint8))
    bits = list(bits)

    n = len(bits)
    if patternlen != None:
        m = patternlen
    else:
        m = int(math.floor(math.log(n, 2))) - 2

        if m < 4:
            print("Error. Not enough data for m to be 4")
            return False, 0, None
        m = 4

    # Step 1
    padded_bits = bits + bits[0:m - 1]

    # Step 2
    psi_sq_m = psi_sq_mv1(m, n, padded_bits)
    psi_sq_mm1 = psi_sq_mv1(m - 1, n, padded_bits)
    psi_sq_mm2 = psi_sq_mv1(m - 2, n, padded_bits)

    delta1 = psi_sq_m - psi_sq_mm1
    delta2 = psi_sq_m - (2 * psi_sq_mm1) + psi_sq_mm2

    P1 = gammaincc(2 ** (m - 2), delta1 / 2.0)
    P2 = gammaincc(2 ** (m - 3), delta2 / 2.0)

    success = (P1 >= 0.01) and (P2 >= 0.01)
    return [P1, P2]


def serial_analysis(dir, block_num, save_path):
    file_count = 0

    for root, dirs, files in os.walk(dir):
        for file in files:
            file_count += 1

    q_values = []

    for i in range(0, block_num):
        file_path = dir + '/chunk_{}.bin'.format(i + 1)
        with open(file_path, "rb") as file:
            content = file.read()
        q_value = serial_test(content)
        for j in q_value:
            q_values.append(round(j, 2))
    counter = Counter(q_values)
    fig, ax = plt.subplots(figsize=(8, 6))
    x = list(counter.keys())
    y = list(counter.values())
    ax.bar(x, y, width=0.01, edgecolor='black')
    save_path = save_path + '/serial.png'
    plt.savefig(save_path)


if __name__ == "__main__":
    file_path = '../dataset/ciphertext/4kb/AES_ECB/chunk_5.bin'
    with open(file_path, "rb") as file:
        content = file.read()
    bits = content[0:128000]

    success, _, plist = serial_test(bits)

    print("success =", success)
    print("plist = ", plist)