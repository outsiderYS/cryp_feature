import os
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt

from gamma_functions import *


# RANDOM EXCURSION TEST
# 注意输出为一个p-value list，其中包括8个p-value
def random_excursion_test(bits):
    bits = np.unpackbits(np.frombuffer(bits, dtype=np.uint8))
    bits = list(bits)

    n = len(bits)

    x = list()  # Convert to +1,-1
    for bit in bits:
        # if bit == 0:
        x.append((bit * 2) - 1)

    # print "x=",x
    # Build the partial sums
    pos = 0
    s = list()
    for e in x:
        pos = pos + e
        s.append(pos)
    sprime = [0] + s + [0]  # Add 0 on each end

    # print "sprime=",sprime
    # Build the list of cycles
    pos = 1
    cycles = list()
    while (pos < len(sprime)):
        cycle = list()
        cycle.append(0)
        while sprime[pos] != 0:
            cycle.append(sprime[pos])
            pos += 1
        cycle.append(0)
        cycles.append(cycle)
        pos = pos + 1

    J = len(cycles)
    print("J=" + str(J))

    vxk = [['a', 'b', 'c', 'd', 'e', 'f'] for y in [-4, -3, -2, -1, 1, 2, 3, 4]]

    # Count Occurances
    for k in range(6):
        for index in range(8):
            mapping = [-4, -3, -2, -1, 1, 2, 3, 4]
            x = mapping[index]
            cyclecount = 0
            # count how many cycles in which x occurs k times
            for cycle in cycles:
                oc = 0
                # Count how many times x occurs in the current cycle
                for pos in cycle:
                    if (pos == x):
                        oc += 1
                # If x occurs k times, increment the cycle count
                if (k < 5):
                    if oc == k:
                        cyclecount += 1
                else:
                    if k == 5:
                        if oc >= 5:
                            cyclecount += 1
            vxk[index][k] = cyclecount

    # Table for reference random probabilities
    pixk = [[0.5, 0.25, 0.125, 0.0625, 0.0312, 0.0312],
            [0.75, 0.0625, 0.0469, 0.0352, 0.0264, 0.0791],
            [0.8333, 0.0278, 0.0231, 0.0193, 0.0161, 0.0804],
            [0.875, 0.0156, 0.0137, 0.012, 0.0105, 0.0733],
            [0.9, 0.01, 0.009, 0.0081, 0.0073, 0.0656],
            [0.9167, 0.0069, 0.0064, 0.0058, 0.0053, 0.0588],
            [0.9286, 0.0051, 0.0047, 0.0044, 0.0041, 0.0531]]

    success = True
    plist = list()
    for index in range(8):
        mapping = [-4, -3, -2, -1, 1, 2, 3, 4]
        x = mapping[index]
        chisq = 0.0
        for k in range(6):
            top = float(vxk[index][k]) - (float(J) * (pixk[abs(x) - 1][k]))
            top = top * top
            bottom = J * pixk[abs(x) - 1][k]
            chisq += top / bottom
        p = gammaincc(5.0 / 2.0, chisq / 2.0)
        plist.append(p)
        if p < 0.01:
            err = " Not Random"
            success = False
        else:
            err = ""
        print("x = %1.0f\tchisq = %f\tp = %f %s" % (x, chisq, p, err))
    if (J < 500):
        print("J too small (J < 500) for result to be reliable")
    elif success:
        print("PASS")
    else:
        print("FAIL: Data not random")
    return plist


def random_excursion_analysis(dir, block_num, save_path):
    file_count = 0

    for root, dirs, files in os.walk(dir):
        for file in files:
            file_count += 1

    q_values = []

    for i in range(0, block_num):
        file_path = dir + '/chunk_{}.bin'.format(i + 1)
        with open(file_path, "rb") as file:
            content = file.read()
        q_value = random_excursion_test(content)
        for j in q_value:
            q_values.append(round(j, 2))
    counter = Counter(q_values)
    fig, ax = plt.subplots(figsize=(8, 6))
    x = list(counter.keys())
    y = list(counter.values())
    ax.bar(x, y, width=0.01, edgecolor='black')
    save_path = save_path + '/random_excursion.png'
    plt.savefig(save_path)


if __name__ == "__main__":
    file_path = '../dataset/ciphertext/4kb/AES_ECB/total.bin'
    with open(file_path, "rb") as file:
        content = file.read()
    bits = content[0:128000]

    success, _, plist = random_excursion_test(bits)

    print("success =", success)
    print("plist = ", plist)