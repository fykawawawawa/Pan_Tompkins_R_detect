import wfdb
import matplotlib.pyplot as plt

def showSignalAtr(signal,atr=None, marker='*'):
    plt.plot(signal)
    plt.title("ECG Signal")
    if atr is not None:
        for index in atr:
            plt.scatter(index, signal[index], marker=marker)
    plt.show()




if __name__ == '__main__':
    path = "./mit-bih-arrhythmia-database-1.0.0/100"
    sampfrom = 360
    sampto = 3600
    record = wfdb.rdrecord(path, # 文件所在路径
                           sampfrom=sampfrom, # 读取100这个记录的起点，从第0个点开始读
                           sampto=sampto, # 读取记录的终点，到1000个点结束
                           physical=False, # 若为True则读取原始信号p_signal，如果为False则读取数字信号d_signal，默认为False
                           channel_names=['MLII']) # 读取那个通道，也可以用channel_names指定某个通道;如channel_names=['MLII']

    signal = record.d_signal[0:sampto-sampfrom]

    fs = record.fs

    signal_ann = wfdb.rdann(path,
                            "atr",
                            sampfrom=sampfrom,
                            sampto=sampto,
                            shift_samps=True
                            )
    atr = signal_ann.sample

    showSignalAtr(signal,atr)