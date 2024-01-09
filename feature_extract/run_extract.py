import math
import os
from collections import Counter
from fractions import Fraction

from matplotlib import pyplot as plt

# from scipy.special import gamma, gammainc, gammaincc
from gamma_functions import *
import numpy as np
import cmath
import random


def count_ones_zeroes(bits):
    ones = 0
    zeroes = 0
    for bit in bits:
        if (bit == 1):
            ones += 1
        else:
            zeroes += 1
    return (zeroes, ones)


def runs_test(bits):
    bits = np.unpackbits(np.frombuffer(bits, dtype=np.uint8))
    bits = list(bits)

    n = len(bits)
    zeroes, ones = count_ones_zeroes(bits)

    prop = float(ones) / float(n)

    tau = 2.0 / math.sqrt(n)

    if abs(prop - 0.5) > tau:
        return 0

    vobs = 1.0
    for i in range(n - 1):
        if bits[i] != bits[i + 1]:
            vobs += 1.0

    p = math.erfc(abs(vobs - (2.0 * n * prop * (1.0 - prop))) / (2.0 * math.sqrt(2.0 * n) * prop * (1 - prop)))
    success = (p >= 0.01)
    return p


def runs_analysis(dir, block_num, save_path):
    file_count = 0

    for root, dirs, files in os.walk(dir):
        for file in files:
            file_count += 1

    q_values = []

    for i in range(0, block_num):
        file_path = dir + '/chunk_{}.bin'.format(i + 1)
        with open(file_path, "rb") as file:
            content = file.read()
        q_value = runs_test(content)
        q_values.append(round(q_value, 2))
        print(i)
    counter = Counter(q_values)
    fig, ax = plt.subplots(figsize=(8, 6))
    x = list(counter.keys())
    y = list(counter.values())
    ax.bar(x, y, width=0.01, edgecolor='black')
    save_path = save_path + '/runs.png'
    plt.savefig(save_path)


if __name__ == "__main__":
    cipher_mods = ['AES_ECB', 'IDEA_ECB', 'RSA', 'TRIPLE_DES_ECB']
    block_size = 4
    block_num = 500
    for cipher_mod in cipher_mods:
        dir_path = '../dataset/ciphertext/{}kb/'.format(block_size) + cipher_mod
        save_path = '../analysis/{}kb/'.format(block_size) + cipher_mod
        runs_analysis(dir_path, block_num, save_path)
        print(cipher_mod)
