import multiprocessing
import os
import random
import time
from text_create import aes_encrypt, idea_encrypt, triple_des_encrypt, sm4_encrypt


text_block_kb = 4
text_block_size = text_block_kb * 1024
key_64 = "22a1d4c4"
key_128 = "22a1d4c4263e83d7"
key_256 = "22a1d4c4263e83d7f8c33a321eb19ae7"
key_512 = "22a1d4c4263e83d7f8c33a321eb19ae722a1d4c4263e83d7f8c33a321eb19ae7"


def block_encrypt_select(data, algorithm, key, mode='ECB'):
    blocks = []
    block_num = int(len(data) / text_block_size)
    for i in range(0, block_num):
        block = data[i * text_block_size: (i + 1) * text_block_size]
        blocks.append(block)

    encrypt_content = bytes()

    if algorithm == 'AES':
        for i in range(0, len(blocks)):
            chunk = aes_encrypt(key, mode, blocks[i])
            encrypt_content += chunk
        return encrypt_content
    elif algorithm == 'IDEA':
        for i in range(0, len(blocks)):
            chunk = idea_encrypt(key, mode, blocks[i])
            encrypt_content += chunk
        return encrypt_content
    elif algorithm == 'TRIPLE_DES':
        for i in range(0, len(blocks)):
            chunk = triple_des_encrypt(key, mode, blocks[i])
            encrypt_content += chunk
        return encrypt_content
    elif algorithm == 'SM4':
        for i in range(0, len(blocks)):
            chunk = sm4_encrypt(key, mode, blocks[i])
            encrypt_content += chunk
        return encrypt_content


def generate_hex_string(length):
    hex_chars = '0123456789abcdef'
    return ''.join(random.choice(hex_chars) for _ in range(length))


def encrypt_data_with_diff_key(crypt_type, crypt_mod, ratio):
    start_time = time.time()
    file_path = './dataset/raw_data/THUCNews/科技.txt'
    random.seed(100)
    with open(file_path, 'rb') as f:
        content = f.read()
    block_size = int(len(content) * ratio)
    blocks = int(len(content)/block_size)
    save_path = './stft/encrypt_data/{}/{}/'.format(ratio, crypt_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(0, blocks):
        binary_string = generate_hex_string(16)
        encrpt_data = block_encrypt_select(content[i*block_size:(i+1)*block_size], crypt_type, binary_string, crypt_mod)
        file_name = '{}/{}.bin'.format(save_path, i)
        with open(file_name, 'wb') as f:
            f.write(encrpt_data)
    end_time = time.time()
    print("{}_{} png finished! Time cost: {} s".format(crypt_type, crypt_mod, end_time - start_time))


if __name__ == '__main__':
    crypt_types = ['AES', 'IDEA', 'TRIPLE_DES', 'SM4']
    crypt_mod = 'ECB'
    pool = multiprocessing.Pool()
    for crypt_type in crypt_types:
        pool.apply_async(encrypt_data_with_diff_key, args=(crypt_type, crypt_mod, 0.01))
        print("{}_{} png create multiprocess start!".format(crypt_type, crypt_mod))
    pool.close()
    pool.join()
    print("Finished!")

