from snownlp import sentiment

# --- 配置参数 ---
POS_FILE = 'data/pos.txt'       # 标注好的积极评论文件
NEG_FILE = 'data/neg.txt'       # 标注好的消极评论文件
MODEL_FILE = 'hsr3.8.marshal' # 训练好的新模型保存路径

def train_new_model():
    """使用标注好的文件训练并保存一个新的情感分析模型。"""
    print("开始训练新的 SnowNLP 模型...")
    print(f"使用积极样本: {POS_FILE}")
    print(f"使用消极样本: {NEG_FILE}")
    
    try:
        sentiment.train(NEG_FILE, POS_FILE)
        print(f"\n训练完成！")
        
        sentiment.save(MODEL_FILE)
        print(f"新模型已成功保存到 -> {MODEL_FILE}")
        
        print("\n--- 如何使用新模型 ---")
        print("在您的分析代码中，请在使用 SnowNLP 前加入以下两行：")
        print("from snownlp import SnowNLP")
        print("from snownlp import sentiment")
        print(f"sentiment.load('{MODEL_FILE}')")
        print("\n示例：")
        print("s = SnowNLP('这条评论真不错')")
        print("print(s.sentiments) # 这将使用您的新模型进行预测")

    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
        print("请检查 pos.txt 和 neg.txt 文件是否存在且包含内容。")

if __name__ == '__main__':
    train_new_model()