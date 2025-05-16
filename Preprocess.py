import numpy as np
import wfdb
from showSignalAtr import showSignalAtr


class Preprocess():
    def __init__(self):
        self.moving_wind = None
        self.square = None
        self.derivative = None
        self.bandpass = None

    def band_pass_filter(self, signal):
        result = None
        sig = signal.copy()

        # low pass filter
        # y(nT) = 2y(nT - T) - y(nT - 2T) + x(nT) - 2x(nT - 6T) + x(nT - 12T)
        for index in range(len(signal)):
            sig[index] = signal[index]
            if index >= 1:
                sig[index] += 2 * sig[index - 1]
            if index >= 2:
                sig[index] -= sig[index - 2]
            if index >= 6:
                sig[index] -= 2 * signal[index - 6]
            if index >= 12:
                sig[index] += signal[index - 12]

        result = sig.copy()

        # high pass filter
        # y(nT) = 32x(nT - 16T) - y(nT - T) - x(nT) + x(nT - 32T)
        for index in range(len(signal)):
            result[index] = -1 * sig[index]
            if index >= 1:
                result[index] -= result[index - 1]
            if index >= 16:
                result[index] += 32 * sig[index - 16]
            if index >= 32:
                result[index] += sig[index - 32]

        # normalization
        result = result / max(max(result), -min(result))
        return result

    def derivative_filter(self, signal):
        result = signal.copy()

        # derivative filter
        # y(nT) = [-x(nT - 2T) - 2x(nT - T) + 2x(nT + T) + x(nT + 2T)]/(8T)

        for index in range(len(signal)):
            result[index] = 0
            if index >= 1:
                result[index] -= 2 * signal[index - 1]
            if index >= 2:
                result[index] -= signal[index - 2]
            if index >= 2 and index + 2 < len(signal):
                result[index] += signal[index + 2]
            if index >= 2 and index + 1 < len(signal):
                result[index] += 2 * signal[index + 1]

        return result

    def squaring(self, signal):
        result = signal.copy()

        for index in range(len(signal)):
            result[index] = signal[index] ** 2

        return result

    def moving_window(self, signal, fs):
        result = signal.copy()
        N = round(0.150 * fs)
        sum = 0
        # first N points
        for index in range(N):
            sum += signal[index] / N
            result[index] = sum

        # y(nT) = [x(nT - (N-1)T) + x(nT - (N-2)T) + ... + x(nT)]/N
        for index in range(N, len(signal)):
            sum += signal[index] / N
            sum -= signal[index - N] / N
            result[index] = sum

        return result

    def solve(self, signal, fs):
        input_signal = signal.copy()
        self.bandpass = self.band_pass_filter(input_signal)
        self.derivative = self.derivative_filter(self.bandpass)
        square = self.squaring(self.derivative)
        self.moving_wind = self.moving_window(square, fs)
        return self.moving_wind


if __name__ == '__main__':
    path = "./mit-bih-arrhythmia-database-1.0.0/100"

    sampfrom = 12000
    sampto = 13500
    # record 650000,1    fs=360
    record = wfdb.rdrecord(path,  # 文件所在路径
                           sampfrom=sampfrom,  # 读取100这个记录的起点，从第0个点开始读
                           sampto=sampto,  # 读取记录的终点，到1000个点结束
                           physical=False,  # 若为True则读取原始信号p_signal，如果为False则读取数字信号d_signal，默认为False
                           channel_names=['MLII'])  # 读取那个通道，也可以用channel_names指定某个通道;如channel_names=['MLII']

    signal = record.d_signal[0:sampto - sampfrom]
    fs = record.fs

    signal_ann = wfdb.rdann(path,
                            "atr",
                            sampfrom=sampfrom,
                            sampto=sampto,
                            shift_samps=True
                            )
    atr = signal_ann.sample

    Preprocess = Preprocess()
    mwin = Preprocess.solve(signal, fs)
    showSignalAtr(Preprocess.bandpass, atr)
    showSignalAtr(Preprocess.moving_wind, atr)
