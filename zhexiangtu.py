import matplotlib.pyplot as plt

# 定义 x 轴数据
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 定义 y 轴数据
y = [3, 5, 9, 7, 6, 8, 10, 12, 8, 11]

# 绘制蓝色折线图，关键点用小圆点表示
plt.plot(x, y, marker='o', color='blue')

# 添加标题和标签
plt.title('折线图示例')
plt.xlabel('X轴标签')
plt.ylabel('Y轴标签')

# 显示网格
plt.grid(True)