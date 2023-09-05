import math
import os
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt


# RANDOM EXCURSION VARIANT TEST
# 注意输出为一个p-value list,其中包含18个p-value
def random_excursion_variant_test(bits):
    bits = np.unpackbits(np.frombuffer(bits, dtype=np.uint8))
    bits = list(bits)

    n = len(bits)

    x = list()  # Convert to +1,-2
    for bit in bits:
        x.append((bit * 2) - 1)

    # Build the partial sums
    pos = 0
    s = list()
    for e in x:
        pos = pos + e
        s.append(pos)
    sprime = [0] + s + [0]  # Add 0 on each end

    # Count the number of cycles J
    J = 0
    for value in sprime[1:]:
        if value == 0:
            J += 1
    print("J=", J)
    # Build the counts of offsets
    count = [0 for x in range(-9, 10)]
    for value in sprime:
        if (abs(value) < 10):
            count[value] += 1

    # Compute P values
    success = True
    plist = list()
    for x in range(-9, 10):
        if x != 0:
            top = abs(count[x] - J)
            bottom = math.sqrt(2.0 * J * ((4.0 * abs(x)) - 2.0))
            p = math.erfc(top / bottom)
            plist.append(p)
            if p < 0.01:
                err = " Not Random"
                success = False
            else:
                err = ""
            print("x = %1.0f\t count=%d\tp = %f %s" % (x, count[x], p, err))

    if (J < 500):
        print("J too small (J=%d < 500) for result to be reliable" % J)
    elif success:
        print("PASS")
    else:
        print("FAIL: Data not random")
    return plist


def random_excursion_variant_analysis(dir, block_num, save_path):
    file_count = 0

    for root, dirs, files in os.walk(dir):
        for file in files:
            file_count += 1

    q_values = []

    for i in range(0, block_num):
        file_path = dir + '/chunk_{}.bin'.format(i + 1)
        with open(file_path, "rb") as file:
            content = file.read()
        q_value = random_excursion_variant_test(content)
        for j in q_value:
            q_values.append(round(j, 2))
    counter = Counter(q_values)
    fig, ax = plt.subplots(figsize=(8, 6))
    x = list(counter.keys())
    y = list(counter.values())
    ax.bar(x, y, width=0.01, edgecolor='black')
    save_path = save_path + '/random_excursion_variant.png'
    plt.savefig(save_path)


if __name__ == "__main__":
    file_path = '../dataset/ciphertext/4kb/AES_ECB/total.bin'
    with open(file_path, "rb") as file:
        content = file.read()
    bits = content[0:128000]

    success, _, plist = random_excursion_variant_test(bits)

    print("success =", success)
    print("plist = ", plist)