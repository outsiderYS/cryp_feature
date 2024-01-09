import tkinter

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.ticker import MaxNLocator
from statistics import mean

import tensorflow as tf
from tensorflow import keras
from text_create import *
from feature_extract import approximate_entropy_feature_extract, binary_matrix_rank_feature_extract
from feature_extract import cumulative_sums_feature_extract, dft_feature_extract, frequency_feature_extract
from feature_extract import frequency_within_block_feature_extract, linear_complexity_feature_extract
from feature_extract import longest_run_ones_in_a_block_feature_extract, maurers_universal_feature_extract
from feature_extract import non_overlapping_template_matching_feature_extract
from feature_extract import overlapping_template_matching_feature_extract
from feature_extract import random_excursion_feature_extract, random_excursion_variant_feature_extract
from feature_extract import run_extract, serial_feature_extract
from stft import bit_process
import random

from PIL import ImageTk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from tkinter.ttk import Combobox
from torchvision.models.densenet import DenseNet
import torchvision.transforms as transforms
from PIL import Image


file_encoder = {'AES_ECB': 0, 'IDEA_ECB': 1, 'TRIPLE_DES_ECB': 2, 'RSA': 3}
num_encrypt = {0: 'AES_ECB', 1: 'IDEA_ECB', 2: 'SM4_ECB', 3: 'TRIPLE_DES_ECB'}
pretrained_model = './recognition/model/best.pth.tar'
device = torch.device('cpu')
model_num_labels = 4


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """

    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint)
    else:
        # this helps avoid errors when loading single-GPU-trained weights onto CPU-model
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


class RandomEncryptFileGeneratorApp:

    # 初始化函数，创建图形界面元素
    def __init__(self, rt):
        # 设置主窗口
        self.root = rt

        # 创建标签和输入框用于输入文件大小
        self.size_label = Label(rt, text="文件大小 (KB):")
        self.size_label.grid(row=0, column=0)
        self.size_entry = Entry(rt)
        self.size_entry.grid(row=1, column=0)

        # 算法选择
        self.encrypt_label = Label(rt, text="选择使用的加密算法:")
        self.encrypt_label.grid(row=0, column=1)
        self.selectEncrypt = StringVar()
        self.comb_encrypt = Combobox(rt, textvariable=self.selectEncrypt, values=['AES', 'IDEA', 'TRIPLE_DES',
                                                                                  'RSA', 'SM2', 'SM4'])
        self.comb_encrypt.grid(row=1, column=1)

        # 加密模式选择
        self.encrypt_mod_label = Label(rt, text="选择使用的加密模式:")
        self.encrypt_mod_label.grid(row=0, column=2)
        self.selectMod = StringVar()
        self.comb_mod = Combobox(rt, textvariable=self.selectMod, values=['ECB', 'CBC', 'CFB', 'OFB', 'CTR'])
        self.comb_mod.grid(row=1, column=2)
        self.encrypt_notice_label = Label(rt, text="SM4仅支持ECB和CBC，仅AES支持CTR")
        self.encrypt_notice_label.grid(row=1, column=3, columnspan=5)

        # 创建按钮，点击按钮触发生成文件操作
        self.generate_button = Button(rt, text="生成加密随机文件", command=self.generate_random_file)
        self.generate_button.grid(row=2, column=0)

        # 文件选择按钮，点击按钮触发选择文件操作
        self.file_select_button = Button(rt, text="选择文件", command=self.file_select)
        self.file_select_button.grid(row=2, column=1)

        # 文件加密按钮，点击按钮触发文件加密
        self.file_encry_button = Button(rt, text="文件加密", command=self.file_encry)
        self.file_encry_button.grid(row=2, column=2)

        # 创建一个属性用于存储生成的文件内容
        self.generated_content = bytes()  # 初始化为空的 bytes 对象

        self.encrypt_content = bytes()

    # 生成随机文件的函数
    def generate_random_file(self):
        try:

            # 获取用户输入的文件大小
            size_kb = int(self.size_entry.get())

            # 如果输入的大小不合法或未选择路径，则退出
            if size_kb <= 0:
                return

            if self.selectEncrypt.get() == '':
                return

            if self.selectMod.get() == '' and self.selectEncrypt.get() == 'RSA' and self.selectMod.get() == 'SM2':
                return

            # 生成随机内容并存储在属性中
            self.generated_content = bytes([random.randint(0, 255) for _ in range(size_kb * 1024)])

            self.encrypt_content = encrypt_select(self.generated_content, self.selectEncrypt.get(),
                                                  self.selectMod.get())

            # 显示成功消息框
            messagebox.showinfo("生成成功", f"成功生成随机加密数据")

        # 捕获输入大小不合法的错误
        except ValueError:
            # 显示错误消息框
            messagebox.showerror("错误", "请进行有效输入")

    def file_select(self):
        file_name = filedialog.askopenfilename()
        if file_name != '':
            with open(file_name, "rb") as file:
                content = file.read()
            self.encrypt_content = content
            # 显示成功消息框
            messagebox.showinfo("选择成功", f"成功选择文件")
        else:
            return

    def file_encry(self):
        try:

            # 获取用户输入的文件大小
            size_kb = len(self.encrypt_content)

            # 如果输入的大小不合法或未选择路径，则退出
            if size_kb <= 0:
                return

            if self.selectEncrypt.get() == '':
                return

            if self.selectMod.get() == '' and self.selectEncrypt.get() == 'RSA' and self.selectMod.get() == 'SM2':
                return

            self.encrypt_content = encrypt_select(self.encrypt_content, self.selectEncrypt.get(),
                                                  self.selectMod.get())

            # 显示成功消息框
            messagebox.showinfo("加密成功", f"成功生成加密数据")

        # 捕获输入大小不合法的错误
        except ValueError:
            # 显示错误消息框
            messagebox.showerror("错误", "请进行有效输入")


class SelectTest(object):

    def __init__(self, rt):
        self.lb_select = Label(rt, text="选择所需的测试", fg='black', font=("黑体", 20))
        self.lb_select.grid(row=3, column=0, columnspan=2)
        self.approximate_entropy = IntVar()
        self.binary_matrix_rank = IntVar()
        self.cumulative_sum = IntVar()
        self.dft = IntVar()
        self.freq = IntVar()
        self.freq_block = IntVar()
        self.linear = IntVar()
        self.longest_run = IntVar()
        self.maurers = IntVar()
        self.non_temp = IntVar()
        self.over_temp = IntVar()
        self.rand_exc = IntVar()
        self.rand_exc_var = IntVar()
        self.run = IntVar()
        self.serial = IntVar()
        self.ch1 = Checkbutton(rt, text='近似熵测试', variable=self.approximate_entropy, onvalue=1, offvalue=0)
        self.ch1.grid(row=5, column=0)
        self.ch2 = Checkbutton(rt, text='二元矩阵秩测试', variable=self.binary_matrix_rank, onvalue=1, offvalue=0)
        self.ch2.grid(row=5, column=1)
        self.ch3 = Checkbutton(rt, text='累加和测试', variable=self.cumulative_sum, onvalue=1, offvalue=0)
        self.ch3.grid(row=5, column=2)
        self.ch4 = Checkbutton(rt, text='离散傅里叶变换测试', variable=self.dft, onvalue=1, offvalue=0)
        self.ch4.grid(row=5, column=3)
        self.ch5 = Checkbutton(rt, text='频率测试', variable=self.freq, onvalue=1, offvalue=0)
        self.ch5.grid(row=5, column=4)
        self.ch6 = Checkbutton(rt, text='块内频数测试', variable=self.freq_block, onvalue=1, offvalue=0)
        self.ch6.grid(row=6, column=0)
        self.ch7 = Checkbutton(rt, text='线性复杂度测试', variable=self.linear, onvalue=1, offvalue=0)
        self.ch7.grid(row=6, column=1)
        self.ch8 = Checkbutton(rt, text='块内最长游程测试', variable=self.longest_run, onvalue=1, offvalue=0)
        self.ch8.grid(row=6, column=2)
        self.ch9 = Checkbutton(rt, text='Maurers的通用统计测试', variable=self.maurers, onvalue=1, offvalue=0)
        self.ch9.grid(row=6, column=3)
        self.ch10 = Checkbutton(rt, text='非重叠模块匹配测试', variable=self.non_temp, onvalue=1, offvalue=0)
        self.ch10.grid(row=6, column=4)
        self.ch11 = Checkbutton(rt, text='重叠模块匹配测试', variable=self.over_temp, onvalue=1, offvalue=0)
        self.ch11.grid(row=7, column=0)
        self.ch12 = Checkbutton(rt, text='随机游走测试', variable=self.rand_exc, onvalue=1, offvalue=0)
        self.ch12.grid(row=7, column=1)
        self.ch13 = Checkbutton(rt, text='随机游走状态频数测试', variable=self.rand_exc_var, onvalue=1, offvalue=0)
        self.ch13.grid(row=7, column=2)
        self.ch14 = Checkbutton(rt, text='游程测试', variable=self.run, onvalue=1, offvalue=0)
        self.ch14.grid(row=7, column=3)
        self.ch15 = Checkbutton(rt, text='线性复杂度测试', variable=self.serial, onvalue=1, offvalue=0)
        self.ch15.grid(row=7, column=4)

    def empty(self):
        if self.approximate_entropy.get() == 0 and self.binary_matrix_rank.get() == 0 and \
                self.cumulative_sum.get() == 0 and self.dft.get() == 0 and self.freq.get() == 0 and \
                self.freq_block.get() == 0 and self.linear.get() == 0 and self.longest_run.get() == 0 and \
                self.maurers.get() == 0 and self.non_temp.get() == 0 and self.over_temp.get() == 0 and \
                self.rand_exc.get() == 0 and self.rand_exc_var.get() == 0 and self.run.get() == 0 and \
                self.serial.get() == 0:
            return True
        else:
            return False


def msg_show(title, msg):
    answer = messagebox.askokcancel(title, msg)


def divide(data, size):
    blocks = []
    text_block_size = size * 1024
    block_num = int(len(data) / text_block_size)
    for i in range(0, block_num):
        block = data[i * text_block_size: (i + 1) * text_block_size]
        blocks.append(block)
    return blocks


def imageWindow(name, data, rt, rg):
    winNew = Toplevel(rt)
    winNew.geometry('1080x800+374+182')
    alg = rg.selectEncrypt.get()
    mod = rg.selectMod.get()
    if alg == 'RSA' or alg == 'SM2':
        mod = ''
    winNew.title('{}-{}的{}结果'.format(alg, mod, name))
    x = [i for i in range(1, 101)]
    y = data

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x[0:10], y[0:10], width=1, edgecolor='black')
    plt.hlines(0.05, 0, 10, colors='red')
    plt.title("Top 10 file blocks")
    plt.xlabel('file block number')
    plt.ylabel('p-value')
    canvas_10 = FigureCanvasTkAgg(fig, master=winNew)
    canvas_10.get_tk_widget().grid(row=0, column=0)
    canvas_10.draw()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x[0:20], y[0:20], width=1, edgecolor='black')
    plt.hlines(0.05, 0, 20, colors='red')
    plt.title("Top 20 file blocks")
    plt.xlabel('file block number')
    plt.ylabel('p-value')
    canvas_20 = FigureCanvasTkAgg(fig, master=winNew)
    canvas_20.get_tk_widget().grid(row=0, column=1)
    canvas_20.draw()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x[0:50], y[0:50], width=1, edgecolor='black')
    plt.hlines(0.05, 0, 50, colors='red')
    plt.title("Top 50 file blocks")
    plt.xlabel('file block number')
    plt.ylabel('p-value')
    canvas_50 = FigureCanvasTkAgg(fig, master=winNew)
    canvas_50.get_tk_widget().grid(row=1, column=0)
    canvas_50.draw()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x[0:100], y[0:100], width=1, edgecolor='black')
    plt.hlines(0.05, 0, 100, colors='red')
    plt.title("Top 100 file blocks")
    plt.xlabel('file block number')
    plt.ylabel('p-value')
    canvas_100 = FigureCanvasTkAgg(fig, master=winNew)
    canvas_100.get_tk_widget().grid(row=1, column=1)
    canvas_100.draw()


class StartTest(object):

    def __init__(self, rt, rg, st):
        self.rg = rg
        self.st = st
        self.rt = rt
        self.lb_block = Label(rt, text='分块大小（kb）:', fg='black', font=("黑体", 10))
        self.lb_block.grid(row=8, column=0)
        self.selectSize = IntVar()
        self.comb_size = Combobox(rt, textvariable=self.selectSize, values=[4, 16, 64])
        self.comb_size.grid(row=8, column=1)
        self.lb_notice = Label(rt, text='分块后块数需大于100', fg='black', font=("黑体", 10))
        self.lb_notice.grid(row=8, column=2)
        self.button = Button(rt, text="开始测试", font=("宋体", 25), fg="black", command=self.start)
        self.button.grid(row=9, column=0)
        self.var = StringVar()
        self.lb_result = Label(rt, textvariable=self.var, fg='black', font=("黑体", 10))
        self.lb_result.grid(row=10, column=0, columnspan=5)
        self.blocks = []
        self.p_values = []

    def start(self):
        if not self.rg.encrypt_content:
            msg_show('警告', '未生成密文!')
            return
        if self.st.empty() is True:
            msg_show('警告', '未选择所需测试！')
            return
        if self.selectSize.get() == 0:
            msg_show('警告', '未选择分块大小！')
            return
        self.blocks = divide(self.rg.encrypt_content, self.selectSize.get())
        if self.st.freq.get() == 1:
            self.p_values = [frequency_feature_extract.frequency_test(self.blocks[i]) for i in range(0, 100)]
            imageWindow('频率测试', self.p_values, self.rt, self.rg)
        if self.st.approximate_entropy.get() == 1:
            self.p_values = [approximate_entropy_feature_extract.approximate_entropy_test(self.blocks[i]) for i in
                             range(0, 100)]
            imageWindow('近似熵测试', self.p_values, self.rt, self.rg)
        if self.st.freq_block.get() == 1:
            self.p_values = [frequency_within_block_feature_extract.frequency_within_block_test(self.blocks[i]) for i in
                             range(0, 100)]
            imageWindow('块内频数测试', self.p_values, self.rt, self.rg)
        if self.st.run.get() == 1:
            self.p_values = [run_extract.runs_test(self.blocks[i]) for i in
                             range(0, 100)]
            imageWindow('游程测试', self.p_values, self.rt, self.rg)
        if self.st.non_temp.get() == 1:
            self.p_values = [mean(non_overlapping_template_matching_feature_extract. \
                                  non_overlapping_template_matching_test(self.blocks[i], temp_select=0))
                             for i in range(0, 100)]
            imageWindow('非重叠模块匹配测试', self.p_values, self.rt, self.rg)
        if self.st.dft.get() == 1:
            self.p_values = [dft_feature_extract.dft_test(self.blocks[i]) for i in range(0, 100)]
            imageWindow('离散傅里叶变换测试', self.p_values, self.rt, self.rg)
        if self.st.longest_run.get() == 1:
            self.p_values = [longest_run_ones_in_a_block_feature_extract.longest_run_ones_in_a_block_test \
                             (self.blocks[i]) for i in range(0, 100)]
            imageWindow('块内最长游程测试', self.p_values, self.rt, self.rg)
        if self.st.binary_matrix_rank.get() == 1:
            self.p_values = [binary_matrix_rank_feature_extract.binary_matrix_rank_test(self.blocks[i], M=28, Q=28)
                             for i in range(0, 100)]
            imageWindow('二元矩阵秩测试', self.p_values, self.rt, self.rg)
        if self.st.over_temp.get() == 1:
            self.p_values = [overlapping_template_matching_feature_extract.overlapping_template_matching_test \
                             (self.blocks[i], blen=6) for i in range(0, 100)]
            imageWindow('重叠模块匹配测试', self.p_values, self.rt, self.rg)
        if self.st.maurers.get() == 1:
            self.p_values = [maurers_universal_feature_extract.maurers_universal_test(self.blocks[i], patternlen=2, \
                                                                                      initblocks=2)
                             for i in range(0, 100)]
            imageWindow('Maurer通用统计测试', self.p_values, self.rt, self.rg)
        if self.st.linear.get() == 1:
            self.p_values = [linear_complexity_feature_extract.linear_complexity_test(self.blocks[i], patternlen=256)
                             for i in range(0, 100)]
            imageWindow('线性复杂度测试', self.p_values, self.rt, self.rg)
        if self.st.serial.get() == 1:
            self.p_values = [mean(serial_feature_extract.serial_test(self.blocks[i]))
                             for i in range(0, 100)]
            imageWindow('序列测试', self.p_values, self.rt, self.rg)
        if self.st.cumulative_sum.get() == 1:
            self.p_values = [mean(cumulative_sums_feature_extract.cumulative_sums_test(self.blocks[i]))
                             for i in range(0, 100)]
            imageWindow('累加和测试', self.p_values, self.rt, self.rg)
        if self.st.rand_exc.get() == 1:
            self.p_values = [mean(random_excursion_feature_extract.random_excursion_test(self.blocks[i]))
                             for i in range(0, 100)]
            imageWindow('随机游走测试', self.p_values, self.rt, self.rg)
        if self.st.rand_exc_var.get() == 1:
            self.p_values = [mean(random_excursion_variant_feature_extract.random_excursion_variant_test(self.blocks[i]))
                             for i in range(0, 100)]
            imageWindow('随机游走状态频数测试', self.p_values, self.rt, self.rg)

        print("success")


class StartRec(object):
    def __init__(self, rt, rg, st):
        self.rg = rg
        self.rt = rt
        self.st = st
        self.lb_block = Label(rt, text='加密算法识别', fg='black', font=("黑体", 10))
        self.lb_block.grid(row=11, column=0)
        self.button = Button(rt, text="开始", font=("宋体", 25), fg="black", command=self.start)
        self.button.grid(row=12, column=0)
        self.model = DenseNet(growth_rate=32, block_config=(6, 12, 24, 16),
                              num_init_features=64, bn_size=4, drop_rate=0.3, num_classes=model_num_labels).to(device)

    def start(self):
        if not self.rg.encrypt_content:
            msg_show('警告', '未生成密文!')
            return
        if self.st.selectSize.get() == 0:
            msg_show('警告', '未选择分块大小！')
            return
        encrypt = self.rg.encrypt_content
        step = self.st.selectSize.get() * 1024
        load_checkpoint(pretrained_model, self.model)
        self.model.eval()

        png_matrix = bit_process.png_maker(encrypt[0: 224 * 224], 224)
        im = Image.fromarray(png_matrix.astype(np.uint8))
        # im.save('./temp/temp.png')

        transform = transforms.Compose([
            transforms.Resize(size=(224, 224), interpolation=3),
            transforms.ToTensor(),
        ])

        png_tensor = transform(im)
        png_tensor = torch.unsqueeze(png_tensor, dim=0)

        y_pred = self.model(png_tensor)
        y_pred = y_pred[0].tolist()

        max_value = max(y_pred)  # 获取最大值
        max_index = y_pred.index(max_value)  # 获取最大值所在的位置

        y = num_encrypt[max_index]

        msg = "该密文使用的加密算法为：" + y
        msg_show("识别结果", msg)


if __name__ == '__main__':
    # 创建窗口：实例化一个窗口对象。
    root = Tk()
    # 窗口大小
    root.geometry("800x600+374+182")
    #  窗口标题
    root.title("随机性测试")

    # 文件选择控件
    fileGenerator = RandomEncryptFileGeneratorApp(root)
    # 测试选择控件
    selectTest = SelectTest(root)
    # 测试开始控件
    startTest = StartTest(root, fileGenerator, selectTest)
    # 识别开始控件
    startRec = StartRec(root, fileGenerator, startTest)
    # 显示窗口
    root.mainloop()
