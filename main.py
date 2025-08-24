import sys
print(sys.executable)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.signal import butter, filtfilt
#butter는 butterworth filter(low-pass, high-pass, band-pass)
#filtfilt는 zero-phase filtering을 해줌.

def moving_average(data, window_size=5):
    window = np.ones(int(window_size))/ float(window_size)
    return np.convolve(data, window, mode = 'same')

def butter_lowpass_filter(data, cutoff, fs, order=2):
    nyq = 0.5 * fs #샘플링 frequency의 절반이 nyquist frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False) #필터 계수 계산
    return filtfilt(b, a, data)  # 앞뒤로 필터링을 해서 phase delay를 없애줌.

class Kalman1D:
    def __init__(self, q=1e-4, r=0.1, x0=0.0, p0=1.0):
        self.q = q  #시스템이 바뀌는 정도를 예상한 값.
        self.r = r  # 측정값의 노이즈 정도를 예상한 값.
        self.x = x0 # 초기 추정값
        self.p = p0 # 초기 오차 공분산

    def update(self, z):
        # Predict
        self.p = self.p + self.q
        # Kalman gain
        k = self.p / (self.p + self.r)
        # Update estimate with measurement z
        self.x = self.x + k * (z - self.x)
        # Update error covariance
        self.p = (1 - k) * self.p
        return self.x



# -------------------------------------------------------------------------
# Generate clean data
fs = 50.0         # samples per second
duration = 10.0   # seconds
n = int(fs * duration)
t = np.linspace(0, duration, n, endpoint=False)
clean = 1.0 * np.sin(2 * np.pi * 1.0 * t)
# sine wave with 1 Hz frequency
# Add Gaussian noise
np.random.seed(42)
noise_std = 0.35
signal = clean + np.random.normal(0, noise_std, size=clean.shape)

#moving average
ma = moving_average(signal, window_size=7)

#Butterworth low-pass
cutoff = 2.5
lp = butter_lowpass_filter(signal, cutoff, fs, order=2)

# Kalman 1D (use observed signal; if using real CSV without "clean", this still works)
noise_variance_guess = np.var(signal - moving_average(signal, 7))
kf = Kalman1D(q=1e-4, r=max(noise_variance_guess, 1e-6), x0=signal[0], p0=1.0)
kf_out = np.array([kf.update(z) for z in signal])



# OUTPUT TABLE
# ---------------------------------------------------------------------------
df_out = pd.DataFrame({
    "time": t,
    "signal_noisy": signal,
    "moving_avg_w7": ma,
    "butter_lp_2.5Hz": lp,
    "kalman_1d": kf_out
})

# If we generated synthetic data, also include the clean reference:
df_out["signal_clean"] = clean
df_out.to_csv("filtered_results.csv", index=False)
df_out.to_excel("filtered_results.xlsx", index=False)  # requires openpyxl



# PLOT
# -----------------------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(t, signal, label="Noisy (input)", alpha=0.45)
plt.plot(t, ma, label="Moving Avg (w=7)", linewidth=1.5)
plt.plot(t, lp, label="Butterworth LP (fc=2.5 Hz)", linewidth=1.5)
plt.plot(t, kf_out, label="Kalman 1D", linewidth=1.5)
plt.plot(t, clean, label="Clean (truth)", linestyle="--", linewidth=1.2)
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (arb. units)")
plt.title("Sensor Data Filtering — Before vs After")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("filter_comparison.png", dpi=200)
plt.show()