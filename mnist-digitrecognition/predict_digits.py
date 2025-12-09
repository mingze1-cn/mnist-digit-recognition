"""导入库"""
import matplotlib
import matplotlib.pyplot as plt
import warnings
import cv2
import numpy as np
import tensorflow as tf
import os

# 强制 TkAgg 后端
matplotlib.use('TkAgg')
# 防止中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
# 忽略警告
warnings.filterwarnings('ignore')

# 创建预测数字函数
def predict_digits(image_path, model):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 判断图像是否存在
    if image is None:
        print('无法加载图像，请检查路径是否有误！')

        return None, None, None
    
    # 固定图像大小
    if image.shape != (28, 28):
        image = cv2.resize(image, (28, 28))
    
    # 归一化处理
    image = image / 255

    # 颜色翻转
    if np.mean(image) > 0.5:
        image = 1 - image
    
    # 创建批次数据
    img_batch = np.array([image])
    # 预测
    prediction = model.predict(img_batch)

    # 获取预测数字
    number = np.argmax(prediction)
    # 获取置信度
    confidence = np.max(prediction)

    return number, confidence, image

# 加载模型
model = tf.keras.models.load_model('mnist_model.keras')
# 图像编号
img_idx = 1
# 预测计数器
predict_count = 0

while True:
    # 获取图像路径
    image = f'digits/digit{img_idx}.png'

    # 判断图像是否存在
    if not os.path.isfile(image):
        break

    try:
        print(f'处理：{image}')

        # 调用预测函数
        num, acc, img = predict_digits(image, model)

        # 打印结果
        print(f'预测结果：{num}')
        print(f'置信度：{acc:.2%}')

        # 设置显示图像
        plt.imshow(img, 'gray')
        # 设置标题
        plt.title(f'预测结果：{num} 置信度：{acc:.2%}')
        plt.axis('off')
        plt.show(block=False)
        plt.pause(3)
        plt.close()

        # 预测数加一
        predict_count += 1

    except Exception as e:
        print(f'错误：{e}')

    # 图像编号加一
    img_idx += 1

# 打印处理总数
print(f'共处理{predict_count}张图像！')