import tkinter as tk
from tkinter import filedialog,messagebox
import os
import random

import numpy as np
import tensorflow as tf

# 独立脚本运行
if __name__ == "__main__":

    num_encrypt = {0: 'AES_ECB', 1: 'IDEA_ECB', 2: 'TRIPLE_DES_ECB', 3: 'RSA'}
    Y = num_encrypt[1]
    print('success')