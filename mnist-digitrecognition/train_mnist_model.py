"""导入库"""
import warnings
import tensorflow as tf

# 忽略警告
warnings.filterwarnings('ignore')

# 加载手写数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 归一化处理
x_train = x_train / 255
x_test = x_test / 255

# 创建神经网络模型
model = tf.keras.models.Sequential()

# 创建第一卷积块
model.add(tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D(2))

# 创建第二卷积块
model.add(tf.keras.layers.Conv2D(64, 3, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2))

# 创建第三卷积块
model.add(tf.keras.layers.Conv2D(128, 3, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2))

# 添加展平层
model.add(tf.keras.layers.GlobalAveragePooling2D())
# 添加全连接层
model.add(tf.keras.layers.Dense(128, activation='relu'))
# 添加dropout层
model.add(tf.keras.layers.Dropout(0.5))
# 添加分类层
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# 编译模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
# 训练模型
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# 获取差值
acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
gap = acc - val_acc
print(f'差值：{gap:.2%}')

# 评估模型
_, test_acc = model.evaluate(x_test, y_test)
print(f'准确率：{test_acc:.2%}')

# 保存模型
model.save('mnist_model.keras')