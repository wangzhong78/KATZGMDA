import numpy as np

# 打开原始文件和目标文件
with open('input.txt', 'r') as input_file, open('output.txt', 'w') as output_file:
    # 逐行读取原始文件中的数字，并进行处理后写入目标文件
    for line in input_file:
        numbers = line.strip().split()  # 假设每行只有一个数字
        processed_numbers = [str(int(number) - 1) for number in numbers]  # 将数字-1并转换为字符串
        output_file.write(' '.join(processed_numbers) + '\n')  # 写入目标文件


with open('input.txt', 'r') as input_file, open('output.txt', 'w') as output_file:
    line = input_file.read()  # 读取整行数据
    numbers = line.strip().split()  # 假设每个数字之间用空格分隔
    processed_numbers = [str(int(number) - 1) for number in numbers]  # 将数字-1并转换为字符串
    output_file.write(' '.join(processed_numbers))  # 写入目标文件，不需要加换行符