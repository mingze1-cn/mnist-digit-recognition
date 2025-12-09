# Chinese-Sentiment-Analysis  
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.10+](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://tensorflow.org)

基于 **Word2Vec + 双向 LSTM** 的中文评论情感分类工具。  
训练数据：豆瓣/京东正负评论样本（`positive_samples.md` / `negative_samples.md`）  
验证集准确率 ≈ **92%**（10 epoch，Adam，128 batch）。

---

## ⚠️ 第一步：下载中文词向量（必须手动完成）
文件较大（≈ 300 MB），**GitHub 不托管**，请自行下载：  
[sgns.zhihu.bigram.bz2](https://pan.baidu.com/s/1pKLcfkR9iJGCz5BjMh2JMQ)（提取码：abcd）  
下载后**保持文件名**放入本地路径：  

data/embeddings/sgns.zhihu.bigram.bz2
复制


---

## 🚀 快速开始
```bash
# 1. 克隆仓库
git clone https://github.com/your_name/chinese-sentiment-analysis.git
cd chinese-sentiment-analysis

# 2. 安装依赖
pip install -r requirements.txt

# 3. 训练模型（自动生成检查点、图向量、日志）
python language_processing.py

# 4. 单句预测
python -c "from language_processing import predict_sentiment; predict_sentiment('房间很凉爽，空调冷气很足')"

📁 项目结构
复制

chinese-sentiment-analysis
├── language_processing.py          # 主脚本（训练+预测）
├── data
│   ├── embeddings/                 # 词向量目录（需下载）
│   ├── positive_samples.md         # 好评训练样本
│   ├── negative_samples.md         # 差评训练样本
│   └── sentiment_checkpoint.keras  # 训练后检查点（自动生成）
├── requirements.txt                # 依赖列表
└── README.md                       # 本文件

📦 主要依赖

    Python ≥ 3.8
    tensorflow ≥ 2.10
    gensim ≥ 4.0
    jieba ≥ 0.42
    scikit-learn ≥ 1.1
    matplotlib ≥ 3.5

📊 训练日志示例
复制

Epoch 10/10
loss: 0.0846 - accuracy: 0.9682 - val_loss: 0.1521 - val_accuracy: 0.9234

🔗 相关链接

    中文词向量项目
    jieba 分词

📄 许可证
MIT License —— 可自由商用、修改，请注明出处。
💬 反馈
欢迎提 Issue / PR，顺手给个 ⭐ 就更好了！
