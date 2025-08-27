import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Tkagg')
# 生成数据
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# 绘制等高线图
plt.contourf(X, Y, Z, cmap='coolwarm')
plt.colorbar()
plt.title('2D等高线密度图')
plt.xlabel('X轴')
plt.ylabel('Y轴')
plt.show()