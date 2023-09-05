import math
import os
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt

import gf2matrix


# 若想在输入为4kb的情况下使用，需调整M和Q的大小使得M*Q*38<4096*8
def binary_matrix_rank_test(bits, M=32, Q=32):
    bits = np.unpackbits(np.frombuffer(bits, dtype=np.uint8))
    bits = list(bits)

    n = len(bits)
    N = int(math.floor(n / (M * Q)))  # Number of blocks

    if N < 38:
        print("  Number of blocks must be greater than 37")
        p = 0.0
        return False, p, None

    # Compute the reference probabilities for FM, FMM and remainder
    r = M
    product = 1.0
    for i in range(r):
        upper1 = (1.0 - (2.0 ** (i - Q)))
        upper2 = (1.0 - (2.0 ** (i - M)))
        lower = 1 - (2.0 ** (i - r))
        product = product * ((upper1 * upper2) / lower)
    FR_prob = product * (2.0 ** ((r * (Q + M - r)) - (M * Q)))

    r = M - 1
    product = 1.0
    for i in range(r):
        upper1 = (1.0 - (2.0 ** (i - Q)))
        upper2 = (1.0 - (2.0 ** (i - M)))
        lower = 1 - (2.0 ** (i - r))
        product = product * ((upper1 * upper2) / lower)
    FRM1_prob = product * (2.0 ** ((r * (Q + M - r)) - (M * Q)))

    LR_prob = 1.0 - (FR_prob + FRM1_prob)

    FM = 0  # Number of full rank matrices
    FMM = 0  # Number of rank -1 matrices
    remainder = 0
    for blknum in range(N):
        block = bits[blknum * (M * Q):(blknum + 1) * (M * Q)]
        # Put in a matrix
        matrix = gf2matrix.matrix_from_bits(M, Q, block, blknum)
        # Compute rank
        rank = gf2matrix.rank(M, Q, matrix, blknum)

        if rank == M:  # count the result
            FM += 1
        elif rank == M - 1:
            FMM += 1
        else:
            remainder += 1

    chisq = (((FM - (FR_prob * N)) ** 2) / (FR_prob * N))
    chisq += (((FMM - (FRM1_prob * N)) ** 2) / (FRM1_prob * N))
    chisq += (((remainder - (LR_prob * N)) ** 2) / (LR_prob * N))
    p = math.e ** (-chisq / 2.0)
    success = (p >= 0.01)

    return p


def binary_matrix_rank_analysis(dir, block_num, save_path):
    file_count = 0

    for root, dirs, files in os.walk(dir):
        for file in files:
            file_count += 1

    q_values = []

    for i in range(0, block_num):
        file_path = dir + '/chunk_{}.bin'.format(i + 1)
        with open(file_path, "rb") as file:
            content = file.read()
        q_value = binary_matrix_rank_test(content, M=16, Q=16)
        q_values.append(round(q_value, 2))
    counter = Counter(q_values)
    fig, ax = plt.subplots(figsize=(8, 6))
    x = list(counter.keys())
    y = list(counter.values())
    ax.bar(x, y, width=0.01, edgecolor='black')
    save_path = save_path + '/binary_matrix_rank.png'
    plt.savefig(save_path)


if __name__ == "__main__":
    file_path = '../dataset/ciphertext/4kb/AES_ECB/total.bin'
    with open(file_path, "rb") as file:
        content = file.read()
    bits = content[0:128000]

    s1, s2, s3 = binary_matrix_rank_test(bits)
    print(s1)
    print("p value is %s" % s2)
