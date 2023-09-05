import os
import numpy as np
from scipy.stats import binom_test
from scipy.fft import fft, fftfreq
from scipy.special import erfc
import matplotlib.pyplot as plt
from collections import Counter


def frequency_test(data):
    binary_data = data

    zero_bits = 0
    one_bits = 0

    # 遍历二进制数据的每个字节和每个位
    for byte in binary_data:
        for bit in range(8):
            # 检查当前位是否为 0 或 1
            if (byte >> bit) & 1 == 0:
                zero_bits += 1
            else:
                one_bits += 1

    # 计算0和1的比例
    total_bits = len(binary_data) * 8

    p_value_0 = binom_test(zero_bits, n=total_bits, p=0.5)

    return p_value_0


def frequency_analysis(dir, block_num, save_path):
    file_count = 0

    for root, dirs, files in os.walk(dir):
        for file in files:
            file_count += 1

    q_values = []

    for i in range(0, block_num):
        file_path = dir + '/chunk_{}.bin'.format(i + 1)
        with open(file_path, "rb") as file:
            content = file.read()
        q_value = frequency_test(content)
        q_values.append(round(q_value, 2))
    counter = Counter(q_values)
    fig, ax = plt.subplots(figsize=(8, 6))
    x = list(counter.keys())
    y = list(counter.values())
    ax.bar(x, y, width=0.01, edgecolor='black')
    save_path = save_path + '/frequency.png'
    plt.savefig(save_path)


if __name__ == '__main__':
    cipher_mods = ['AES_ECB', 'IDEA_ECB', 'RSA', 'TRIPLE_DES_ECB']
    block_size = 64
    block_num = 500
    for cipher_mod in cipher_mods:
        dir_path = '../dataset/ciphertext/{}kb/'.format(block_size) + cipher_mod
        save_path = '../analysis/{}kb/'.format(block_size) + cipher_mod
        frequency_analysis(dir_path, block_num, save_path)
