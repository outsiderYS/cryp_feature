import math
import os
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt


def dft_test(bits):
    bits = np.unpackbits(np.frombuffer(bits, dtype=np.uint8))
    bits = list(bits)

    n = len(bits)
    if (n % 2) == 1:  # Make it an even number
        bits = bits[:-1]

    ts = list()  # Convert to +1,-1
    for bit in bits:
        ts.append((bit * 2) - 1)

    ts_np = np.array(ts)
    fs = np.fft.fft(ts_np)  # Compute DFT

    mags = abs(fs)[:round(n / 2)]  # Compute magnitudes of first half of sequence

    T = math.sqrt(math.log(1.0 / 0.05) * n)  # Compute upper threshold
    N0 = 0.95 * n / 2.0

    N1 = 0.0  # Count the peaks above the upper theshold
    for mag in mags:
        if mag < T:
            N1 += 1.0
    d = (N1 - N0) / math.sqrt((n * 0.95 * 0.05) / 4)  # Compute the P value
    p = math.erfc(abs(d) / math.sqrt(2))

    success = (p >= 0.01)
    return p


def dft_analysis(dir, block_num, save_path):
    file_count = 0

    for root, dirs, files in os.walk(dir):
        for file in files:
            file_count += 1

    q_values = []

    for i in range(0, block_num):
        file_path = dir + '/chunk_{}.bin'.format(i + 1)
        with open(file_path, "rb") as file:
            content = file.read()
        q_value = dft_test(content)
        q_values.append(round(q_value, 2))
        print(i)
    counter = Counter(q_values)
    fig, ax = plt.subplots(figsize=(8, 6))
    x = list(counter.keys())
    y = list(counter.values())
    ax.bar(x, y, width=0.01, edgecolor='black')
    save_path = save_path + '/dft.png'
    plt.savefig(save_path)


if __name__ == "__main__":
    cipher_mods = ['AES_ECB', 'IDEA_ECB', 'RSA', 'TRIPLE_DES_ECB']
    block_size = 4
    block_num = 500
    for cipher_mod in cipher_mods:
        dir_path = '../dataset/ciphertext/{}kb/'.format(block_size) + cipher_mod
        save_path = '../analysis/{}kb/'.format(block_size) + cipher_mod
        dft_analysis(dir_path, block_num, save_path)
        print(cipher_mod)
