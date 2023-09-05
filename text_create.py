import os
import random

from cryptography.hazmat.primitives import padding, hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric import padding as asy_padding
from gmssl import sm2, sm3, func
from gmssl.sm4 import CryptSM4, SM4_ENCRYPT, SM4_DECRYPT

file_path = 'dataset/raw_data/Sword_of_Dawn.txt'
text_block_kb = 4
text_block_size = text_block_kb * 1024
key_64 = "22a1d4c4"
key_128 = "22a1d4c4263e83d7"
key_256 = "22a1d4c4263e83d7f8c33a321eb19ae7"
key_512 = "22a1d4c4263e83d7f8c33a321eb19ae722a1d4c4263e83d7f8c33a321eb19ae7"
save_path = "dataset/ciphertext/{}kb".format(text_block_kb)
iv = key_128.encode('utf-8')
siv = key_64.encode('utf-8')


def text_divide():
    with open(file_path, "rb") as file:
        content = file.read()
    blocks = []
    block_num = int(len(content)/text_block_size)
    for i in range(0, block_num):
        block = content[i*text_block_size: (i+1)*text_block_size]
        blocks.append(block)
    return blocks


def rsa_text_divide():
    with open(file_path, "rb") as file:
        content = file.read()
    blocks = []
    block_num = int(len(content)/245)
    for i in range(0, block_num):
        block = content[i*245: (i+1)*245]
        blocks.append(block)
    return blocks


def aes_encrypt(secret_key, mode, data):
    """加密数据
    :param secret_key: 加密秘钥-256bit
    :param mode: 加密模式 ECB,CBC,CFB,OFB,CTR
    :param data: 需要加密数据
    """
    # 将数据转换为byte类型
    secret_key = secret_key.encode("utf-8")

    # # 填充数据采用pkcs7
    # padder = padding.PKCS7(cipher_block_size).padder()
    # pad_data = padder.update(data)
    # pad_data += padder.finalize()

    # 创建密码器
    if mode == 'ECB':
        cipher = Cipher(
            algorithms.AES(secret_key),
            mode=modes.ECB(),
            backend=default_backend()
        )
    elif mode == 'CBC':
        cipher = Cipher(
            algorithms.AES(secret_key),
            mode=modes.CBC(iv),
            backend=default_backend()
        )
    elif mode == 'CFB':
        cipher = Cipher(
            algorithms.AES(secret_key),
            mode=modes.CFB(iv),
            backend=default_backend()
        )
    elif mode == 'OFB':
        cipher = Cipher(
            algorithms.AES(secret_key),
            mode=modes.OFB(iv),
            backend=default_backend()
        )
    elif mode == 'CTR':
        cipher = Cipher(
            algorithms.AES(secret_key),
            mode=modes.CTR(iv),
            backend=default_backend()
        )

    # 加密数据
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(data)
    return encrypted_data


def aes_ciphertext_gen(secret_key, mode):
    text_blocks = text_divide()
    chunk_path = save_path + f"/AES_" + mode
    if not os.path.exists(chunk_path):
        os.makedirs(chunk_path)
    for i in range(0, len(text_blocks)):
        chunk_file_path = chunk_path + f"/chunk_{i + 1}.bin"  # 生成分块文件名
        chunk = aes_encrypt(secret_key, mode, text_blocks[i])
        with open(chunk_file_path, 'wb') as chunk_file:
            chunk_file.write(chunk)
    chunk_file_path = save_path + "/total.bin"  # 生成分块文件名
    with open(chunk_file_path, 'wb') as chunk_file:
        for i in range(0, len(text_blocks)):
            chunk = aes_encrypt(secret_key, mode, text_blocks[i])
            chunk_file.write(chunk)


def idea_encrypt(secret_key, mode, data):
    """加密数据
        :param secret_key: 加密秘钥-128bit
        :param mode: 加密模式 ECB,CBC,CFB,OFB
        :param data: 需要加密数据
        """
    # 将数据转换为byte类型
    secret_key = secret_key.encode("utf-8")

    # # 填充数据采用pkcs7
    # padder = padding.PKCS7(cipher_block_size).padder()
    # pad_data = padder.update(data)
    # pad_data += padder.finalize()

    # 创建密码器
    if mode == 'ECB':
        cipher = Cipher(
            algorithms.IDEA(secret_key),
            mode=modes.ECB(),
            backend=default_backend()
        )
    elif mode == 'CBC':
        cipher = Cipher(
            algorithms.IDEA(secret_key),
            mode=modes.CBC(siv),
            backend=default_backend()
        )
    elif mode == 'CFB':
        cipher = Cipher(
            algorithms.IDEA(secret_key),
            mode=modes.CFB(siv),
            backend=default_backend()
        )
    elif mode == 'OFB':
        cipher = Cipher(
            algorithms.IDEA(secret_key),
            mode=modes.OFB(siv),
            backend=default_backend()
        )
    elif mode == 'CTR':
        cipher = Cipher(
            algorithms.IDEA(secret_key),
            mode=modes.CTR(iv),
            backend=default_backend()
        )

    # 加密数据
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(data)
    return encrypted_data


def idea_ciphertext_gen(secret_key, mode):
    text_blocks = text_divide()
    chunk_path = save_path + f"/IDEA_" + mode
    if not os.path.exists(chunk_path):
        os.makedirs(chunk_path)
    for i in range(0, len(text_blocks)):
        chunk_file_path = chunk_path + f"/chunk_{i + 1}.bin"  # 生成分块文件名
        chunk = idea_encrypt(secret_key, mode, text_blocks[i])
        with open(chunk_file_path, 'wb') as chunk_file:
            chunk_file.write(chunk)
    chunk_file_path = chunk_path + "/total.bin"  # 生成分块文件名
    with open(chunk_file_path, 'wb') as chunk_file:
        for i in range(0, len(text_blocks)):
            chunk = idea_encrypt(secret_key, mode, text_blocks[i])
            chunk_file.write(chunk)


def triple_des_encrypt(secret_key, mode, data):
    """加密数据
        :param secret_key: 加密秘钥-128bit
        :param mode: 加密模式 ECB,CBC,CFB,OFB,CTR
        :param data: 需要加密数据
        """
    # 将数据转换为byte类型
    secret_key = secret_key.encode("utf-8")

    # # 填充数据采用pkcs7
    # padder = padding.PKCS7(cipher_block_size).padder()
    # pad_data = padder.update(data)
    # pad_data += padder.finalize()

    # 创建密码器
    if mode == 'ECB':
        cipher = Cipher(
            algorithms.TripleDES(secret_key),
            mode=modes.ECB(),
            backend=default_backend()
        )
    elif mode == 'CBC':
        cipher = Cipher(
            algorithms.TripleDES(secret_key),
            mode=modes.CBC(siv),
            backend=default_backend()
        )
    elif mode == 'CFB':
        cipher = Cipher(
            algorithms.TripleDES(secret_key),
            mode=modes.CFB(siv),
            backend=default_backend()
        )
    elif mode == 'OFB':
        cipher = Cipher(
            algorithms.TripleDES(secret_key),
            mode=modes.OFB(siv),
            backend=default_backend()
        )
    elif mode == 'CTR':
        cipher = Cipher(
            algorithms.TripleDES(secret_key),
            mode=modes.CTR(iv),
            backend=default_backend()
        )

    # 加密数据
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(data)
    return encrypted_data


def triple_des_ciphertext_gen(secret_key, mode):
    text_blocks = text_divide()
    chunk_path = save_path + f"/TRIPLE_DES_" + mode
    if not os.path.exists(chunk_path):
        os.makedirs(chunk_path)
    for i in range(0, len(text_blocks)):
        chunk_file_path = chunk_path + f"/chunk_{i + 1}.bin"  # 生成分块文件名
        chunk = triple_des_encrypt(secret_key, mode, text_blocks[i])
        with open(chunk_file_path, 'wb') as chunk_file:
            chunk_file.write(chunk)
    chunk_file_path = chunk_path + "/total.bin"  # 生成分块文件名
    with open(chunk_file_path, 'wb') as chunk_file:
        for i in range(0, len(text_blocks)):
            chunk = triple_des_encrypt(secret_key, mode, text_blocks[i])
            chunk_file.write(chunk)


def rsa_encrypt(secret_key, data):
    """加密数据
            :param secret_key: 加密秘钥-public_key
            :param data: 需要加密数据
            """
    ciphertext = secret_key.encrypt(
        data,
        asy_padding.OAEP(
            mgf=asy_padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return ciphertext


def rsa_ciphertext_gen(secret_key):
    text_blocks = rsa_text_divide()
    chunk_path = save_path + f"/RSA"
    if not os.path.exists(chunk_path):
        os.makedirs(chunk_path)
    chunk_file_path = chunk_path + "/total.bin"  # 生成分块文件名
    with open(chunk_file_path, 'wb') as chunk_file:
        for i in range(0, len(text_blocks)):
            chunk = rsa_encrypt(secret_key, text_blocks[i])
            chunk_file.write(chunk)
    i = 0
    with open(chunk_file_path, 'rb') as file:
        while True:
            chunk = file.read(text_block_size)
            if not chunk:
                break
            chunk_file_path = chunk_path + f"/chunk_{i + 1}.bin"  # 生成分块文件名
            with open(chunk_file_path, 'wb') as chunk_file:
                chunk_file.write(chunk)
            i = i + 1


def sm2_encrypt(pub_key, pri_key, data):
    """加密数据
                :param pub_key: 公钥
                :param pri_key: 私钥
                :param data: 需要加密数据
                """
    sm2_crypt = sm2.CryptSM2(public_key=pub_key, private_key=pri_key)
    ciphertext = sm2_crypt.encrypt(data)
    return ciphertext


def sm2_ciphertext_gen(pub_key, pri_key):
    chunk_path = save_path + f"/SM2"
    if not os.path.exists(chunk_path):
        os.makedirs(chunk_path)

    chunk_file_path = chunk_path + "/total.bin"  # 生成分块文件名
    with open(chunk_file_path, 'wb') as chunk_file:
        with open(file_path, "rb") as file:
            content = file.read()
            chunk = sm2_encrypt(pub_key, pri_key, content)
            chunk_file.write(chunk)
    i = 0
    with open(chunk_file_path, 'rb') as file:
        while True:
            chunk = file.read(text_block_size)
            if not chunk:
                break
            chunk_file_path = chunk_path + f"/chunk_{i + 1}.bin"  # 生成分块文件名
            with open(chunk_file_path, 'wb') as chunk_file:
                chunk_file.write(chunk)
            i = i + 1


def sm4_encrypt(secret_key, mode, data):
    """加密数据
            :param secret_key: 加密秘钥-128bit
            :param mode: 加密模式 ECB,CBC
            :param data: 需要加密数据
            """
    # 将数据转换为byte类型
    secret_key = secret_key.encode("utf-8")
    crypt_sm4 = CryptSM4()
    crypt_sm4.set_key(secret_key, SM4_ENCRYPT)
    if mode == 'ECB':
        encrypt_value = crypt_sm4.crypt_ecb(data)
    elif mode == 'CBC':
        encrypt_value = crypt_sm4.crypt_cbc(iv, data)
    else:
        print("没有该加密模式")
    return encrypt_value


def sm4_ciphertext_gen(secret_key, mode):
    """SM4只支持ECB和CBC模式"""
    text_blocks = text_divide()
    chunk_path = save_path + f"/SM4_" + mode
    if not os.path.exists(chunk_path):
        os.makedirs(chunk_path)
    for i in range(0, len(text_blocks)):
        chunk_file_path = chunk_path + f"/chunk_{i + 1}.bin"  # 生成分块文件名
        chunk = sm4_encrypt(secret_key, mode, text_blocks[i])
        with open(chunk_file_path, 'wb') as chunk_file:
            chunk_file.write(chunk)
    chunk_file_path = chunk_path + "/total.bin"  # 生成分块文件名
    with open(chunk_file_path, 'wb') as chunk_file:
        for i in range(0, len(text_blocks)):
            chunk = sm4_encrypt(secret_key, mode, text_blocks[i])
            chunk_file.write(chunk)


def encrypt_select(data, algorithm, mode='ECB'):
    blocks = []
    block_num = int(len(data) / text_block_size)
    for i in range(0, block_num):
        block = data[i * text_block_size: (i + 1) * text_block_size]
        blocks.append(block)

    encrypt_content = bytes()
    if algorithm == 'RSA':
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
        public_key = private_key.public_key()
        blocks = []
        block_num = int(len(data) / 245)
        for i in range(0, block_num):
            block = data[i * 245: (i + 1) * 245]
            blocks.append(block)
        for i in range(0, len(blocks)):
            chunk = rsa_encrypt(public_key, blocks[i])
            encrypt_content += chunk
        return encrypt_content
    elif algorithm == 'SM2':
        sm2_private_key = '67F3A86FDE67EB91A9D39230B90396076BE1B0963E579E626743A0C68914D04C'
        sm2_public_key = 'FE47E4AF6A8C68316ADEB7F0D6EAEE288F310851A2CB2F33D3A828F27753EDCB3' \
                         'CCF37A27EA4DCB77BC8BA827851AC893D309395B9B994029F4097226EFCF8D4'
        encrypt_content = sm2_encrypt(sm2_public_key, sm2_private_key, data)
        return encrypt_content
    elif algorithm == 'AES':
        for i in range(0, len(blocks)):
            chunk = aes_encrypt(key_256, mode, blocks[i])
            encrypt_content += chunk
        return encrypt_content
    elif algorithm == 'IDEA':
        for i in range(0, len(blocks)):
            chunk = idea_encrypt(key_128, mode, blocks[i])
            encrypt_content += chunk
        return encrypt_content
    elif algorithm == 'TRIPLE_DES':
        for i in range(0, len(blocks)):
            chunk = triple_des_encrypt(key_128, mode, blocks[i])
            encrypt_content += chunk
        return encrypt_content
    elif algorithm == 'SM4':
        for i in range(0, len(blocks)):
            chunk = sm4_encrypt(key_128, mode, blocks[i])
            encrypt_content += chunk
        return encrypt_content


if __name__ == '__main__':
    # AES,IDEA,TRIPLE_DES支持ECB,CBC,CFB,OFB模式,其中仅AES可以使用CTR
    # private_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
    # public_key = private_key.public_key()
    # rsa_ciphertext_gen(public_key)
    # triple_des_ciphertext_gen(key_128, 'ECB')
    # idea_ciphertext_gen(key_128, 'ECB')
    # aes_ciphertext_gen(key_256, 'ECB')
    # sm2_private_key = '67F3A86FDE67EB91A9D39230B90396076BE1B0963E579E626743A0C68914D04C'
    # sm2_public_key = 'FE47E4AF6A8C68316ADEB7F0D6EAEE288F310851A2CB2F33D3A828F27753EDCB3' \
    #                  'CCF37A27EA4DCB77BC8BA827851AC893D309395B9B994029F4097226EFCF8D4'
    # sm2_ciphertext_gen(sm2_public_key, sm2_private_key)
    # sm4只支持ECB和CBC模式
    # sm4_ciphertext_gen(key_128, 'ECB')

    size_kb = 400
    # 生成随机内容并存储在属性中

    generated_content = bytes([random.randint(0, 255) for _ in range(size_kb * 1024)])

    result = encrypt_select(generated_content, 'SM2')

    print("success")
