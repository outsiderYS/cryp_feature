import math
import os
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt


def pattern2int(pattern):
    l = len(pattern)
    n = 0
    for bit in (pattern):
        n = (n << 1) + bit
    return n


# 比特长度至少为 387,840，若需使用4kb的输入，则需要修改patternlen和initblocks，例如2和4
def maurers_universal_test(bits, patternlen=None, initblocks=None):
    bits = np.unpackbits(np.frombuffer(bits, dtype=np.uint8))
    bits = list(bits)

    n = len(bits)

    # Step 1. Choose the block size
    if patternlen != None:
        L = patternlen
    else:
        ns = [904960, 2068480, 4654080, 10342400,
              22753280, 49643520, 107560960,
              231669760, 496435200, 1059061760]
        L = 6
        if n < 387840:
            print("Error. Need at least 387840 bits. Got %d." % n)
            exit()
        for threshold in ns:
            if n >= threshold:
                L += 1

                # Step 2 Split the data into Q and K blocks
    nblocks = int(math.floor(n / L))
    if initblocks != None:
        Q = initblocks
    else:
        Q = 10 * (2 ** L)
    K = nblocks - Q

    # Step 3 Construct Table
    nsymbols = (2 ** L)
    T = [0 for x in range(nsymbols)]  # zero out the table
    for i in range(Q):  # Mark final position of
        pattern = bits[i * L:(i + 1) * L]  # each pattern
        idx = pattern2int(pattern)
        T[idx] = i + 1  # +1 to number indexes 1..(2**L)+1
        # instead of 0..2**L
    # Step 4 Iterate
    sum = 0.0
    for i in range(Q, nblocks):
        pattern = bits[i * L:(i + 1) * L]
        j = pattern2int(pattern)
        dist = i + 1 - T[j]
        T[j] = i + 1
        sum = sum + math.log(dist, 2)

    # Step 5 Compute the test statistic
    fn = sum / K

    # Step 6 Compute the P Value
    # Tables from https://static.aminer.org/pdf/PDF/000/120/333/
    # a_universal_statistical_test_for_random_bit_generators.pdf
    ev_table = [0, 0.73264948, 1.5374383, 2.40160681, 3.31122472,
                4.25342659, 5.2177052, 6.1962507, 7.1836656,
                8.1764248, 9.1723243, 10.170032, 11.168765,
                12.168070, 13.167693, 14.167488, 15.167379]
    var_table = [0, 0.690, 1.338, 1.901, 2.358, 2.705, 2.954, 3.125,
                 3.238, 3.311, 3.356, 3.384, 3.401, 3.410, 3.416,
                 3.419, 3.421]

    # sigma = math.sqrt(var_table[L])
    mag = abs((fn - ev_table[L]) / ((math.sqrt(var_table[L])) * math.sqrt(2)))
    P = math.erfc(mag)

    success = (P >= 0.01)
    return P


def maurers_universal_analysis(dir, block_num, save_path):
    file_count = 0

    for root, dirs, files in os.walk(dir):
        for file in files:
            file_count += 1

    q_values = []

    for i in range(0, block_num):
        file_path = dir + '/chunk_{}.bin'.format(i + 1)
        with open(file_path, "rb") as file:
            content = file.read()
        q_value = maurers_universal_test(content, patternlen=2, initblocks=4)
        q_values.append(round(q_value, 2))
    counter = Counter(q_values)
    fig, ax = plt.subplots(figsize=(8, 6))
    x = list(counter.keys())
    y = list(counter.values())
    ax.bar(x, y, width=0.01, edgecolor='black')
    save_path = save_path + '/maurers_universal.png'
    plt.savefig(save_path)


if __name__ == "__main__":
    file_path = '../dataset/ciphertext/4kb/AES_ECB/total.bin'
    with open(file_path, "rb") as file:
        content = file.read()
    bits = content[0:128000]

    success, p, _ = maurers_universal_test(bits, patternlen=7, initblocks=1280)

    print("success =", success)
    print("p       = ", p)