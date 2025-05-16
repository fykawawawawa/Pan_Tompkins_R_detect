import numpy as np
import wfdb
import matplotlib.pyplot as plt
from scipy import signal as sg
from Preprocess import Preprocess
from showSignalAtr import showSignalAtr


class HeartRate():
    def __init__(self, signal, mwin, bpass, fs):
        self.RR1, self.RR2, self.probable_peaks, self.r_locs, self.peaks, self.result = ([] for i in range(6))
        self.SPKI, self.NPKI, self.Threshold_I1, self.Threshold_I2 = (0 for i in range(4))
        self.SPKF, self.NPKF, self.Threshold_F1, self.Threshold_F2 = (0 for i in range(4))

        self.T_wave = False
        self.m_win = mwin
        self.b_pass = bpass
        self.fs = fs
        self.signal = signal
        self.win_150ms = round(0.15 * self.fs)

        self.RR_Low_Limit = 0
        self.RR_High_Limit = 0
        self.RR_Missed_Limit = 0
        self.RR_Average1 = 0

    def approx_peaks(self):
        '''
        smooth filtered signal
        '''
        # moving average filter
        slopes = sg.fftconvolve(self.m_win, np.full((25,), 1) / 25, mode="same")
        # save possible peaks
        for idx in range(self.win_150ms + 1, len(slopes) - 1):
            if slopes[idx] > slopes[idx - 1] and slopes[idx] > slopes[idx + 1]:
                self.peaks.append(idx)

    def adjust_rr_interval(self, ind):
        '''
        Adjust RR interval & limit
        :param ind: current index in self.peaks
        '''

        # Finding the eight most recent RR intervals(time),
        # np.diff求数组的差分 max(0,ind-8):ind+1 have 9 values
        self.RR1 = np.diff(self.peaks[max(0, ind - 8):ind + 1]) / self.fs

        # Calculate RR Average
        self.RR_Average1 = np.mean(self.RR1)
        RR_Average2 = self.RR_Average1

        # Finding the eight most recent RR intervals lying between RR Low Limit and RR High Limit
        if ind >= 8:
            for i in range(0, 8):
                if self.RR_Low_Limit < self.RR1[i] < self.RR_High_Limit:
                    self.RR2.append(self.RR1[i])
                    if len(self.RR2) > 8:
                        self.RR2.remove(self.RR2[0])
                        RR_Average2 = np.mean(self.RR2)

        if len(self.RR2) > 7 or ind < 8:
            self.RR_Low_Limit = 0.92 * RR_Average2
            self.RR_High_Limit = 1.16 * RR_Average2
            self.RR_Missed_Limit = 1.66 * RR_Average2

    def search_back(self, peak_loc, RRn, sb_win):
        '''
        Searchback，当R峰间隔大于丢失限制时，找间隔内大于Threshold_I1的值中的最大值作为一个新的R峰，并更新其他参数
        :param peak_loc: peak location in consideration
        :param RRn: the most recent RR interval
        :param sb_win: searchback window
        '''
        # Check if most recent RR interval is greater than RR_Missed_Limit
        if (RRn > self.RR_Missed_Limit):
            # Initialize a window to searchback
            win_rr = self.m_win[peak_loc - sb_win + 1: peak_loc + 1]
            # Find the x locations inside the window having y values greater than Threshold I1
            coord = np.asarray(win_rr > self.Threshold_I1).nonzero()[0]
            # Find the x location of the max peak value in the search window
            win_max_pos = None
            if len(coord) > 0:
                for pos in coord:
                    if self.m_win[pos] == max(self.m_win[coord]):
                        win_max_pos = pos
                        break
            else:
                win_max_pos = None

            # If the max peak value is found
            if win_max_pos is not None:
                # Update the thresholds corresponding to moving window integration
                # 我们需要moving window integration的阈值，也需要band pass的阈值
                self.SPKI = 0.25 * self.m_win[win_max_pos] + 0.75 * self.SPKI
                self.Threshold_I1 = self.NPKI + 0.25 * (self.SPKI - self.NPKI)
                self.Threshold_I2 = 0.5 * self.Threshold_I1

                # Initialize a window to search back
                win_rr = self.b_pass[peak_loc - sb_win + 1:peak_loc + 1]
                coord = np.asarray(win_rr > self.Threshold_F1).nonzero()[0]
                band_max_pos = None
                if len(coord) > 0:
                    # Find the x locations inside the window having y values greater than Threshold F1
                    for pos in coord:
                        if self.b_pass[pos] == max(self.b_pass[coord]):
                            band_max_pos = pos
                            break
                else:
                    band_max_pos = None
                if band_max_pos is not None:
                    # Update the thresholds corresponding to bandpass filter
                    if self.b_pass[band_max_pos] > self.Threshold_F2:
                        self.SPKF = 0.25 * self.b_pass[band_max_pos] + 0.75 * self.SPKF
                        self.Threshold_F1 = self.NPKF + 0.25 * (self.SPKF - self.NPKF)
                        self.Threshold_F2 = 0.5 * self.Threshold_F1

                        # Append the probable R peak location
                        self.r_locs.append(band_max_pos)

    def find_t_wave(self, peak_loc, RRn, ind, prev_ind):
        '''
        T Wave Identification
        :param peak_loc: peak location in consideration
        :param RRn: the most recent RR interval
        :param ind: current index in peaks array
        :param prev_ind: previous index in peaks array
        '''
        if self.m_win[peak_loc] >= self.Threshold_I1:
            if ind > 0 and 0.20 < RRn < 0.36:
                # Find the slope of current and last waveform detected
                curr_slope = max(np.diff(self.m_win[peak_loc - round(self.win_150ms / 2): peak_loc + 1]))
                last_slope = max(np.diff(self.m_win[self.peaks[prev_ind] - round(self.win_150ms / 2): peak_loc + 1]))
                # If current waveform slope is less than half of last waveform slope
                if curr_slope < 0.5 * last_slope:
                    # T Wave is found and update noise threshold
                    self.T_wave = True
                    # T wave is a noise peak
                    self.NPKI = 0.125 * self.m_win[peak_loc] + 0.875 * self.NPKI

            if (not self.T_wave):
                # T Wave is not found and update signal thresholds
                if self.probable_peaks[ind] > self.Threshold_F1:
                    # it is a signal peak
                    self.SPKI = 0.125 * self.m_win[peak_loc] + 0.875 * self.SPKI
                    self.SPKF = 0.125 * self.b_pass[ind] + 0.875 * self.SPKF

                    # Append the probable R peak location
                    self.r_locs.append(self.probable_peaks[ind])
                else:
                    # for integration waveform, it is a signal peak, but for filtered ECG, it is a noise peak
                    self.SPKI = 0.125 * self.m_win[peak_loc] + 0.875 * self.SPKI
                    self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF

            # Find T Wave, Update noise thresholds
            elif \
                    self.m_win[peak_loc] < self.Threshold_I1 or self.Threshold_I1 < self.m_win[
                        peak_loc] < self.Threshold_I2:
                # Both for integration waveform and filtered ECG, it is a noise peak
                self.NPKI = 0.125 * self.m_win[peak_loc] + 0.875 * self.NPKI
                self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF

    def adjust_thresholds(self, peak_loc, ind):
        '''
        阈值初始化
        Adjust Noise and Signal Thresholds During Learning Phase
        :param peak_loc: peak location in consideration
        :param ind: current index in peaks array
        '''

        if (self.m_win[peak_loc] >= self.Threshold_I1):
            # for integration waveformm, it is a signal peak
            # Update signal threshold
            self.SPKI = 0.125 * self.m_win[peak_loc] + 0.875 * self.SPKI

            if self.probable_peaks[ind] >= self.Threshold_F1:
                self.SPKF = 0.125 * self.b_pass[ind] + 0.875 * self.SPKF

                # Append the probable R peak location
                self.r_locs.append(self.probable_peaks[ind])
            else:
                # Update noise threshold
                self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF

        # Update noise thresholds
        elif (self.m_win[peak_loc] < self.Threshold_I2) or (
                self.Threshold_I2 < self.m_win[peak_loc] < self.Threshold_I1):
            self.NPKI = 0.125 * self.m_win[peak_loc] + 0.875 * self.NPKI
            self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF

    def update_thresholds(self):
        '''
        Update Noise and Signal Thresholds for next iteration
        '''

        self.Threshold_I1 = self.NPKI + 0.25 * (self.SPKI - self.NPKI)
        self.Threshold_F1 = self.NPKF + 0.25 * (self.SPKF - self.NPKF)
        self.Threshold_I2 = 0.5 * self.Threshold_I1
        self.Threshold_F2 = 0.5 * self.Threshold_F1
        self.T_wave = False

    def ecg_searchback(self):
        '''
        Searchback in ECG signal to increase efficiency
        '''

        # Filter the unique R peak locations
        self.r_locs = np.unique(np.array(self.r_locs).astype(int))

        # Initialize a window to searchback
        win_200ms = round(0.2 * self.fs)

        for r_val in self.r_locs:
            coord = np.arange(r_val - win_200ms, min(len(self.signal), r_val + win_200ms + 1), 1)

            x_max = None
            # Find the x location of the max peak value
            if (len(coord) > 0):
                # 找最大值
                for pos in coord:
                    if (self.signal[pos] == max(self.signal[coord])):
                        x_max = pos
                        break
            else:
                x_max = None

            # Append the peak location
            if (x_max is not None):
                self.result.append(x_max)

    def find_r_peaks(self):
        '''
        R Peak Detection
        '''

        # Find approximate peak locations
        self.approx_peaks()

        # Iterate over possible peak locations
        for ind in range(len(self.peaks)):

            # Initialize the search window for peak detection
            peak_loc = self.peaks[ind] # peak location of bpass & mwin
            win_300ms = np.arange(max(0, self.peaks[ind] - self.win_150ms),
                                  min(self.peaks[ind] + self.win_150ms, len(self.b_pass) - 1), 1)
            max_val = max(self.b_pass[win_300ms], default=0)

            # Find the x location of the max peak value
            if max_val != 0:
                x_coord = np.asarray(self.b_pass == max_val).nonzero()
                self.probable_peaks.append(x_coord[0][0]) # nonzero返回元组，[0][0]表示300ms内第一个最大峰的序号

            if ind < len(self.probable_peaks) and ind != 0:
                # Adjust RR interval and limits
                self.adjust_rr_interval(ind)

                # Adjust thresholds in case of irregular beats
                if self.RR_Average1 < self.RR_Low_Limit or self.RR_Average1 > self.RR_Missed_Limit:
                    self.Threshold_I1 /= 2
                    self.Threshold_F1 /= 2

                RRn = self.RR1[-1]

                # Searchback， 检查R峰间隔是否大于丢失限制
                self.search_back(peak_loc, RRn, round(RRn * self.fs))

                # T Wave Identification
                self.find_t_wave(peak_loc, RRn, ind, ind - 1)

            else:
                # Initiate threholds, when ind = 0 or max_val == 0
                self.adjust_thresholds(peak_loc, ind)

            # Update threholds for next iteration
            self.update_thresholds()

        # Searchback in ECG signal
        self.ecg_searchback()

        return np.array(self.result)


if __name__ == '__main__':
    path = "./mit-bih-arrhythmia-database-1.0.0/100"
    sampfrom = 500000
    sampto = 502000
    record = wfdb.rdrecord(path,  # 文件所在路径
                           sampfrom=sampfrom,  # 读取100这个记录的起点，从第0个点开始读
                           sampto=sampto,  # 读取记录的终点，到1000个点结束
                           physical=False,  # 若为True则读取原始信号p_signal，如果为False则读取数字信号d_signal，默认为False
                           channel_names=['MLII'])  # 读取那个通道，也可以用channel_names指定某个通道;如channel_names=['MLII']

    signal = record.d_signal[:, 0:sampto - sampfrom]
    signal = signal.squeeze(-1)
    fs = record.fs # fs=360

    showSignalAtr(signal)

    # 展示算法定位的R峰
    Preprocess = Preprocess()
    # 离散高通滤波后的结果
    bpass = Preprocess.band_pass_filter(signal)
    # 滑动窗口微分滤波后的结果
    mwin = Preprocess.solve(signal, fs)
    HeartRate = HeartRate(signal, mwin, bpass, fs)
    mark = HeartRate.find_r_peaks()
    showSignalAtr(signal, mark,marker='^')


    # 标签的R峰
    signal_ann = wfdb.rdann(path,
                            "atr",
                            sampfrom=sampfrom,
                            sampto=sampto,
                            shift_samps=True
                            )
    atr = signal_ann.sample
    showSignalAtr(signal,atr)