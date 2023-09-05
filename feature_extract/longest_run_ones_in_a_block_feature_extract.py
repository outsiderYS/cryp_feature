import os
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt

from gamma_functions import *


def probs(K, M, i):
    M8 = [0.2148, 0.3672, 0.2305, 0.1875]
    M128 = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
    M512 = [0.1170, 0.2460, 0.2523, 0.1755, 0.1027, 0.1124]
    M1000 = [0.1307, 0.2437, 0.2452, 0.1714, 0.1002, 0.1088]
    M10000 = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
    if (M == 8):
        return M8[i]
    elif (M == 128):
        return M128[i]
    elif (M == 512):
        return M512[i]
    elif (M == 1000):
        return M1000[i]
    else:
        return M10000[i]


def longest_run_ones_in_a_block_test(bits):
    bits = np.unpackbits(np.frombuffer(bits, dtype=np.uint8))
    bits = list(bits)

    n = len(bits)

    if n < 128:
        return 0
    elif n < 6272:
        M = 8
    elif n < 750000:
        M = 128
    else:
        M = 10000

    # compute new values for K & N
    if M == 8:
        K = 3
        N = 16
    elif M == 128:
        K = 5
        N = 49
    else:
        K = 6
        N = 75

    # Table of frequencies
    v = [0, 0, 0, 0, 0, 0, 0]

    for i in range(N):  # over each block
        # find longest run
        block = bits[i * M:((i + 1) * M)]  # Block i

        run = 0
        longest = 0
        for j in range(M):  # Count the bits.
            if block[j] == 1:
                run += 1
                if run > longest:
                    longest = run
            else:
                run = 0

        if M == 8:
            if longest <= 1:
                v[0] += 1
            elif longest == 2:
                v[1] += 1
            elif longest == 3:
                v[2] += 1
            else:
                v[3] += 1
        elif M == 128:
            if longest <= 4:
                v[0] += 1
            elif longest == 5:
                v[1] += 1
            elif longest == 6:
                v[2] += 1
            elif longest == 7:
                v[3] += 1
            elif longest == 8:
                v[4] += 1
            else:
                v[5] += 1
        else:
            if longest <= 10:
                v[0] += 1
            elif longest == 11:
                v[1] += 1
            elif longest == 12:
                v[2] += 1
            elif longest == 13:
                v[3] += 1
            elif longest == 14:
                v[4] += 1
            elif longest == 15:
                v[5] += 1
            else:
                v[6] += 1

    # Compute Chi-Sq
    chi_sq = 0.0
    for i in range(K + 1):
        p_i = probs(K, M, i)
        upper = (v[i] - N * p_i) ** 2
        lower = N * p_i
        chi_sq += upper / lower
    print("  n = " + str(n))
    print("  K = " + str(K))
    print("  M = " + str(M))
    print("  N = " + str(N))
    print("  chi_sq = " + str(chi_sq))
    p = gammaincc(K / 2.0, chi_sq / 2.0)

    success = (p >= 0.01)
    return p


def longest_run_ones_in_a_block_analysis(dir, block_num, save_path):
    file_count = 0

    for root, dirs, files in os.walk(dir):
        for file in files:
            file_count += 1

    q_values = []

    for i in range(0, block_num):
        file_path = dir + '/chunk_{}.bin'.format(i + 1)
        with open(file_path, "rb") as file:
            content = file.read()
        q_value = longest_run_ones_in_a_block_test(content)
        q_values.append(round(q_value, 2))
        print(i)
    counter = Counter(q_values)
    fig, ax = plt.subplots(figsize=(8, 6))
    x = list(counter.keys())
    y = list(counter.values())
    ax.bar(x, y, width=0.01, edgecolor='black')
    save_path = save_path + '/longest_run_ones_in_a_block.png'
    plt.savefig(save_path)


if __name__ == "__main__":
    cipher_mods = ['AES_ECB', 'IDEA_ECB', 'RSA', 'TRIPLE_DES_ECB']
    block_size = 4
    block_num = 500
    for cipher_mod in cipher_mods:
        dir_path = '../dataset/ciphertext/{}kb/'.format(block_size) + cipher_mod
        save_path = '../analysis/{}kb/'.format(block_size) + cipher_mod
        longest_run_ones_in_a_block_analysis(dir_path, block_num, save_path)
        print(cipher_mod)
