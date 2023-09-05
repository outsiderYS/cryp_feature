import math
import os
from collections import Counter
from fractions import Fraction
import numpy as np
from matplotlib import pyplot as plt

from gamma_functions import *


def count_ones_zeroes(bits):
    ones = 0
    zeroes = 0
    for bit in bits:
        if (bit == 1):
            ones += 1
        else:
            zeroes += 1
    return (zeroes, ones)


def frequency_within_block_test(bits):
    # Compute number of blocks M = block size. N=num of blocks
    # N = floor(n/M)
    # miniumum block size 20 bits, most blocks 100
    bits = np.unpackbits(np.frombuffer(bits, dtype=np.uint8))
    bits = list(bits)

    n = len(bits)
    M = 20
    N = int(math.floor(n / M))
    if N > 128:
        N = 128
        M = int(math.floor(n / N))

    if len(bits) < 100:
        print("Too little data for test. Supply at least 100 bits")
        # return (FAIL,1.0,None)

    num_of_blocks = N
    block_size = M  # int(math.floor(len(bits)/num_of_blocks))
    # n = int(block_size * num_of_blocks)

    proportions = list()
    for i in range(num_of_blocks):
        block = bits[i * (block_size):((i + 1) * (block_size))]
        zeroes, ones = count_ones_zeroes(block)
        proportions.append(Fraction(ones, block_size))

    chisq = 0.0
    for prop in proportions:  #
        chisq += 4.0 * block_size * ((prop - Fraction(1, 2)) ** 2)

    p = gammaincc((num_of_blocks / 2.0), float(chisq) / 2.0)
    success = (p >= 0.01)
    return p


def frequency_within_block_analysis(dir, block_num, save_path):
    file_count = 0

    for root, dirs, files in os.walk(dir):
        for file in files:
            file_count += 1

    q_values = []

    for i in range(0, block_num):
        file_path = dir + '/chunk_{}.bin'.format(i + 1)
        with open(file_path, "rb") as file:
            content = file.read()
        q_value = frequency_within_block_test(content)
        q_values.append(round(q_value, 2))
        print(i)
    counter = Counter(q_values)
    fig, ax = plt.subplots(figsize=(8, 6))
    x = list(counter.keys())
    y = list(counter.values())
    ax.bar(x, y, width=0.01, edgecolor='black')
    save_path = save_path + '/frequency_within_block.png'
    plt.savefig(save_path)


if __name__ == "__main__":
    cipher_mods = ['AES_ECB', 'IDEA_ECB', 'RSA', 'TRIPLE_DES_ECB']
    block_size = 4
    block_num = 500
    for cipher_mod in cipher_mods:
        dir_path = '../dataset/ciphertext/{}kb/'.format(block_size) + cipher_mod
        save_path = '../analysis/{}kb/'.format(block_size) + cipher_mod
        frequency_within_block_analysis(dir_path, block_num, save_path)
        print(cipher_mod)