import wfdb
import pywt
import warnings
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as join
from scipy.fftpack import fft
from scipy.signal import filtfilt, butter

warnings.filterwarnings("ignore")


def test(seq):
    """

    :param seq:
    :return:
    """

    ecg_size = 14
    db_size = 8
    record = wfdb.rdrecord(join('training', 'data_{}_{}'.format(seq[0], seq[1])), sampfrom=0, sampto=1000)

    plt.figure(figsize=(ecg_size, 4))
    plt.plot(record.p_signal)
    plt.show()

    data = record.p_signal[:, 0].flatten()

    ###################################################################################################################
    # Butter
    ###################################################################################################################
    [b, a] = butter(5, [0.1/100, 40/100], 'bandpass')
    rdata1 = filtfilt(b, a, data)

    ###################################################################################################################
    # Wavelet-dec
    ###################################################################################################################
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=5)
    coeffs1 = coeffs
    coeffs2 = coeffs

    cA5, cD5, cD4, cD3, cD2, cD1 = coeffs
    obj = [cA5, cD5, cD4, cD3, cD2, cD1]

    f, ax = plt.subplots(6, 1, figsize=(ecg_size, 30))
    for i, com in enumerate(obj):
        index = [
            1 if j == i else 0
            for j in range(len(obj))
        ]

        y = pywt.waverec(np.multiply(coeffs, index).tolist(), 'db5')
        ax[i].plot(y)

    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.95)
    plt.show()

    ###################################################################################################################
    # Wavelet1
    ###################################################################################################################
    p = 0.5
    q = 10

    threshold2 = np.sqrt(np.var(cD1) * np.log(1000))

    for i in range(1, len(coeffs2)):
        w = coeffs2[i]
        update = np.sign(w) * (np.abs(w) - (p * threshold2) / (p + (np.exp(q * (np.abs(w) - threshold2)) - 1)))
        coeffs2[i] = np.where(np.abs(w) > threshold2, update, 0)

    rdata3 = pywt.waverec(coeffs=coeffs2, wavelet='db5')

    ###################################################################################################################
    # Wavelet2
    ###################################################################################################################
    threshold1 = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))

    # 将高频信号cD1、cD2置零
    cD1.fill(0)
    cD2.fill(0)

    # 将其他中低频信号按软阈值公式滤波
    for i in range(1, len(coeffs1) - 2):
        coeffs1[i] = pywt.threshold(coeffs1[i], threshold1)

    rdata2 = pywt.waverec(coeffs=coeffs1, wavelet='db5')

    ###################################################################################################################
    # Plot
    ###################################################################################################################
    plt.figure(figsize=(ecg_size, 4))

    plt.plot(data, label='original')
    plt.plot(rdata1 + np.mean(data), label='butter', alpha=0.8)
    plt.plot(rdata2, label='wavelet-n', alpha=0.8)
    plt.plot(rdata3, label='wavelet-e', alpha=0.8)
    plt.legend()
    plt.show()

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(ecg_size, 18))

    ax1.plot(rdata1 + np.mean(data), label='butter')
    ax1.plot(data, label='original', alpha=0.5)
    ax1.set_title('Butter')
    ax1.legend(loc='upper left')

    ax2.plot(rdata2, label='wavelet-n')
    ax2.plot(data, label='original', alpha=0.5)
    ax2.set_title('Wavelet-n')
    ax2.legend(loc='upper left')

    ax3.plot(rdata3, label='wavelet-e')
    ax3.plot(data, label='original', alpha=0.5)
    ax3.set_title('Wavelet-e')
    ax3.legend(loc='upper left')

    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.95)
    plt.show()

    fdata = np.abs(fft(data))
    fdata1 = np.abs(fft(rdata1))
    fdata2 = np.abs(fft(rdata2))
    fdata3 = np.abs(fft(rdata3))

    length = len(fdata)

    fdata = fdata[range(1, int(length/2))]
    fdata1 = fdata1[range(1, int(length/2))]
    fdata2 = fdata2[range(1, int(length/2))]
    fdata3 = fdata3[range(1, int(length/2))]

    plt.figure(figsize=(db_size, 5))

    plt.plot(fdata, label='original')
    plt.plot(fdata1, label='butter', alpha=0.8)
    plt.plot(fdata2, label='wavelet-n', alpha=0.8)
    plt.plot(fdata3, label='wavelet-e', alpha=0.8)
    plt.legend()
    plt.show()

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(db_size, 15))

    ax1.plot(fdata1, label='butter')
    ax1.plot(fdata, label='original', alpha=0.5)
    ax1.set_title('Butter')
    ax1.legend(loc='upper right')

    ax2.plot(fdata2, label='wavelet-n')
    ax2.plot(fdata, label='original', alpha=0.5)
    ax2.set_title('Wavelet-n')
    ax2.legend(loc='upper right')

    ax3.plot(fdata3, label='wavelet-e')
    ax3.plot(fdata, label='original', alpha=0.5)
    ax3.set_title('Wavelet-e')
    ax3.legend(loc='upper right')

    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.95)
    plt.tight_layout()
    plt.show()

    return 0


def wavelet(data, p=0.5, q=10):
    """
    小波变换函数
    :param data: 要求是单一序列
    :param p: 软值自定义参数
    :param q: 软值自定义参数
    :return:
    """
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=5)
    cA5, cD5, cD4, cD3, cD2, cD1 = coeffs
    threshold = np.sqrt(np.var(cD1) * np.log(len(data)))

    for i in range(1, len(coeffs)):
        w = coeffs[i]
        update = np.sign(w) * (np.abs(w) - (p * threshold) / (p + np.exp(q * (np.abs(w) - threshold)) - 1))
        coeffs[i] = np.where(np.abs(w) > threshold, update, 0)

    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')

    return rdata


if __name__ == '__main__':
    """
    主函数
    
    """
    right = [(70, 5), (58, 2)]
    error0 = [(1, 1)]
    error1 = [(70, 17), (58, 5)]
    error2 = [(64, 8), (68, 16)]
    test(error2[1])

    plt.figure(figsize=(16, 4))
    record = wfdb.rdrecord(join('training', 'data_0_1'), sampfrom=0, sampto=1000)
    data = record.p_signal[:, 0].flatten()

    # example
    data = wavelet(data)
    plt.plot(data, label='wavelet-e')
    plt.plot(record.p_signal[:, 0], label='original', alpha=0.5)
    plt.legend(loc='upper right')
    plt.show()
