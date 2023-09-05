import math
import os
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
from gamma_functions import *


def bits_to_int(bits):
    theint = 0
    for i in range(len(bits)):
        theint = (theint << 1) + bits[i]
    return theint


# 对于不同大小的输入需要调整其中的m值
def approximate_entropy_test(bits):
    bits = np.unpackbits(np.frombuffer(bits, dtype=np.uint8))
    bits = list(bits)

    n = len(bits)

    m = int(math.floor(math.log(n, 2))) - 6
    if m < 2:
        m = 2
    if m > 3:
        m = 3

    Cmi = list()
    phi_m = list()
    for iterm in range(m, m + 2):
        # Step 1
        padded_bits = bits + bits[0:iterm - 1]

        # Step 2
        counts = list()
        for i in range(2 ** iterm):
            # print "  Pattern #%d 0of %d" % (i+1,2**iterm)
            count = 0
            for j in range(n):
                if bits_to_int(padded_bits[j:j + iterm]) == i:
                    count += 1
            counts.append(count)

        # step 3
        Ci = list()
        for i in range(2 ** iterm):
            Ci.append(float(counts[i]) / float(n))

        Cmi.append(Ci)

        # Step 4
        sum = 0.0
        for i in range(2 ** iterm):
            sum += Ci[i] * math.log((Ci[i] / 10.0))
        phi_m.append(sum)

    # Step 5 - let the loop steps 1-4 complete

    # Step 6
    appen_m = phi_m[0] - phi_m[1]
    chisq = 2 * n * (math.log(2) - appen_m)
    # Step 7
    p = gammaincc(2 ** (m - 1), (chisq / 2.0))

    success = (p >= 0.01)
    return p


def approximate_entropy_analysis(dir, block_num, save_path):
    file_count = 0

    for root, dirs, files in os.walk(dir):
        for file in files:
            file_count += 1

    entropy_q_values = []

    for i in range(0, block_num):
        file_path = dir + '/chunk_{}.bin'.format(i + 1)
        with open(file_path, "rb") as file:
            content = file.read()
        q_value = approximate_entropy_test(content)
        entropy_q_values.append(round(q_value, 2))
        print(i)
    counter = Counter(entropy_q_values)
    fig, ax = plt.subplots(figsize=(8, 6))
    x = list(counter.keys())
    y = list(counter.values())
    ax.bar(x, y, width=0.01, edgecolor='black')
    save_path = save_path + '/approximate_entropy.png'
    plt.savefig(save_path)


if __name__ == "__main__":
    cipher_mods = ['AES_ECB', 'IDEA_ECB', 'RSA', 'TRIPLE_DES_ECB']
    block_size = 4
    block_num = 500
    for cipher_mod in cipher_mods:
        dir_path = '../dataset/ciphertext/{}kb/'.format(block_size) + cipher_mod
        save_path = '../analysis/{}kb/'.format(block_size) + cipher_mod
        approximate_entropy_analysis(dir_path, block_num, save_path)
        print(cipher_mod)
