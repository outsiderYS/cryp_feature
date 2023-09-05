import math
import os
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt


def normcdf(n):
    return 0.5 * math.erfc(-n * math.sqrt(0.5))


def p_value(n, z):
    sum_a = 0.0
    startk = int(math.floor((((float(-n) / z) + 1.0) / 4.0)))
    endk = int(math.floor((((float(n) / z) - 1.0) / 4.0)))
    for k in range(startk, endk + 1):
        c = (((4.0 * k) + 1.0) * z) / math.sqrt(n)
        # d = scipy.stats.norm.cdf(c)
        d = normcdf(c)
        c = (((4.0 * k) - 1.0) * z) / math.sqrt(n)
        # e = scipy.stats.norm.cdf(c)
        e = normcdf(c)
        sum_a = sum_a + d - e

    sum_b = 0.0
    startk = int(math.floor((((float(-n) / z) - 3.0) / 4.0)))
    endk = int(math.floor((((float(n) / z) - 1.0) / 4.0)))
    for k in range(startk, endk + 1):
        c = (((4.0 * k) + 3.0) * z) / math.sqrt(n)
        # d = scipy.stats.norm.cdf(c)
        d = normcdf(c)
        c = (((4.0 * k) + 1.0) * z) / math.sqrt(n)
        # e = scipy.stats.norm.cdf(c)
        e = normcdf(c)
        sum_b = sum_b + d - e

    p = 1.0 - sum_a + sum_b
    return p


# 注意输出为一个p-value list，分别为前向和后向的两个p-value
def cumulative_sums_test(bits):
    bits = np.unpackbits(np.frombuffer(bits, dtype=np.uint8))
    bits = list(bits)

    n = len(bits)
    # Step 1
    x = list()  # Convert to +1,-1
    for bit in bits:
        # if bit == 0:
        x.append((bit * 2) - 1)

    # Steps 2 and 3 Combined
    # Compute the partial sum and records the largest excursion.
    pos = 0
    forward_max = 0
    for e in x:
        pos = pos + e
        if abs(pos) > forward_max:
            forward_max = abs(pos)
    pos = 0
    backward_max = 0
    for e in reversed(x):
        pos = pos + e
        if abs(pos) > backward_max:
            backward_max = abs(pos)

    # Step 4
    p_forward = p_value(n, forward_max)
    p_backward = p_value(n, backward_max)

    success = ((p_forward >= 0.01) and (p_backward >= 0.01))
    plist = [p_forward, p_backward]

    if success:
        print("PASS")
    else:
        print("FAIL: Data not random")
    return plist


def cumulative_sums_analysis(dir, block_num, save_path):
    file_count = 0

    for root, dirs, files in os.walk(dir):
        for file in files:
            file_count += 1

    q_values = []

    for i in range(0, block_num):
        file_path = dir + '/chunk_{}.bin'.format(i + 1)
        with open(file_path, "rb") as file:
            content = file.read()
        q_value = cumulative_sums_test(content)
        for j in q_value:
            q_values.append(round(j, 2))
    counter = Counter(q_values)
    fig, ax = plt.subplots(figsize=(8, 6))
    x = list(counter.keys())
    y = list(counter.values())
    ax.bar(x, y, width=0.01, edgecolor='black')
    save_path = save_path + '/cumulative_sums.png'
    plt.savefig(save_path)


if __name__ == "__main__":
    file_path = '../dataset/ciphertext/4kb/AES_ECB/total.bin'
    with open(file_path, "rb") as file:
        content = file.read()
    bits = content[0:128000]

    success, _, plist = cumulative_sums_test(bits)

    print("success =", success)
    print("plist = ", plist)