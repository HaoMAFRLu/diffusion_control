import numpy as np
import matplotlib.pyplot as plt

# def get_gamma(t, a=0.00007, alpha=0.01):
#     return a * (1 - np.exp(-alpha * t))

def get_gamma(t, a=1.0, alpha=0.01, t0=0.5):
    return a / (1 + np.exp(-alpha * (t - t0)))

t_values = np.linspace(0, 500, 500) 
gamma_values = get_gamma(t_values)  

# 绘制图像
plt.figure(figsize=(8, 6))
plt.plot(t_values, gamma_values, label=r"$\gamma(t) = a \cdot (1 - e^{-\alpha \cdot t})$", color="b")
plt.xlabel("t")
plt.ylabel("gamma(t)")
plt.title("Plot of gamma(t) = a * (1 - exp(-alpha * t))")
plt.legend()
plt.grid()
plt.show()
