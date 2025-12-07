import xml.etree.ElementTree as ET
import regex as re  # 注意：这里使用 regex 模块以支持 \p{P} 等高级正则
import os
import jieba
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import random

# ================= 配置区域 =================

# 输入输出配置
XML_FILE_PATH = 'input.xml'          # 弹幕源文件
OUTPUT_FOLDER = 'wc_output'          # 输出目录
FONT_PATH = "D:/Fonts/HarmonyOS Sans SC/HarmonyOS_Sans_SC_Medium.ttf"               # 【重要】请修改为你电脑上的中文字体路径，例如 C:/Windows/Fonts/msyh.ttc
STOPWORDS_PATH = 'wc/stopwords.txt'  # 停用词路径
USER_DICT_PATH = 'wc/user_dict.txt'  # 自定义词典路径
COLOR_IMAGE_PATH = 'wc/color.jpg'    # 颜色提供图片路径（可选，用于提供词云着色）

# 过滤正则 (来自脚本1)
INVALID_REGEX_PATTERNS = [
    r'^(\[[^\]]*\])+$',
    r'^\s*$',
    r'^.*?表情【.*?$',
    r'记忆是梦的开场白',
    r'与.*共舞',
    r'^[\p{P}]+$',      # 需要 regex 模块支持
    r'^6+$',
    r'^[a-zA-Z0-9]+$',
    r'^.*?来了.*?$'
]

# ==========================================

class BiliDanmakuWordCloud:
    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.regex_list = [re.compile(p) for p in INVALID_REGEX_PATTERNS]
        self.raw_texts = []
        self.clean_words = []
        
        # 确保输出目录存在
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)

    def load_stopwords(self):
        """加载停用词"""
        stops = set()
        if os.path.exists(STOPWORDS_PATH):
            with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
                stops = {line.strip() for line in f}
            print(f"已加载停用词库，共 {len(stops)} 个词。")
        else:
            print("警告：未找到停用词文件，将不使用停用词过滤。")
        return stops

    def load_data(self):
        """加载并初步过滤XML数据 (参考脚本1逻辑)"""
        print(f"正在加载 XML 文件: {self.xml_path} ...")
        try:
            tree = ET.parse(self.xml_path)
            root = tree.getroot()
            
            count_total = 0
            count_valid = 0
            
            for child in root.findall('d'):
                text = str(child.text).strip()
                count_total += 1
                
                # 正则过滤
                if self._is_effective(text):
                    self.raw_texts.append(text)
                    count_valid += 1
            
            print(f"数据加载完成。总弹幕: {count_total}, 有效保留: {count_valid}")
            
        except Exception as e:
            print(f"读取XML出错: {e}")
            exit()

    def _is_effective(self, text):
        """正则过滤逻辑"""
        if not text: return False
        for pattern in self.regex_list:
            if pattern.search(text):
                return False
        return True

    def process_text(self):
        """分词与二次过滤 (参考脚本2逻辑)"""
        print("正在进行分词处理...")
        
        # 加载自定义词典
        if os.path.exists(USER_DICT_PATH):
            jieba.load_userdict(USER_DICT_PATH)
            print("已加载自定义词典。")

        stopwords = self.load_stopwords()
        
        # 合并文本进行分词 (也可以逐条分词，这里选择逐条处理以便过滤)
        # 如果数据量巨大，可以考虑多进程，但一般弹幕量级单线程即可
        
        all_words = []
        
        for text in self.raw_texts:
            
            words = jieba.lcut(text)
            for w in words:
                # 过滤逻辑：非停用词 且 长度>1 (去除单字)
                if w not in stopwords and len(w) > 1 and w.strip() != '':
                    all_words.append(w)
        
        self.clean_words = all_words
        print(f"分词完成，有效词汇量: {len(self.clean_words)}")

        # 打印一下Top30方便调试
        c = Counter(self.clean_words)
        print("Top 30 高频词预览:")
        print(c.most_common(30))
        return c

    def extract_colors_from_image(self, image_path, num_colors=100):
        """从图片中提取颜色集合（采样而非按位置映射）"""
        try:
            from PIL import Image
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            
            # 降采样以加快速度
            h, w = img_array.shape[:2]
            sample_h = max(1, h // 20)
            sample_w = max(1, w // 20)
            sampled = img_array[::sample_h, ::sample_w]
            
            # 扁平化并随机采样
            pixels = sampled.reshape(-1, 3)
            if len(pixels) > num_colors:
                indices = np.random.choice(len(pixels), num_colors, replace=False)
                colors = pixels[indices]
            else:
                colors = pixels
            
            # 转换为RGB元组列表
            color_list = [tuple(color) for color in colors]
            return color_list
        except Exception as e:
            print(f"提取颜色失败: {e}")
            return []

    def create_random_color_func(self, color_list):
        """创建随机颜色函数"""
        if not color_list:
            return None
        
        def random_color_func(word=None, font_size=None, position=None, orientation=None, 
                            font_path=None, random_state=None):
            """随机返回颜色列表中的一个颜色"""
            color = random.choice(color_list)
            return 'rgb({},{},{})'.format(color[0], color[1], color[2])
        
        return random_color_func

    def generate_wordcloud(self, word_counts):
        """生成词云图 (参考脚本2逻辑)"""
        print("正在生成词云图...")

        # 配置词云基础参数
        wc_params = {
            "font_path": FONT_PATH,
            "background_color": None,  # 【关键1】背景色设为 None
            "mode": "RGBA",             # 【关键2】模式设为 RGBA 以支持透明
            "stopwords": set(),
            "width": 1920,              # 设置图像宽度
            "height": 1080,             # 设置图像高度
            "max_words": 300,
            "min_word_length": 2,
            "include_numbers": True
        }
        
        # 创建词云对象
        wc = WordCloud(**wc_params)
        # 根据词频生成
        wc.generate_from_frequencies(word_counts)

        # 如果提供了颜色图片，则使用其颜色随机着色
        if os.path.exists(COLOR_IMAGE_PATH):
            try:
                # 从图片中提取颜色（随机采样而非按位置映射）
                colors = self.extract_colors_from_image(COLOR_IMAGE_PATH, num_colors=50)
                if colors:
                    # 使用随机颜色函数重新着色
                    color_func = self.create_random_color_func(colors)
                    wc.recolor(color_func=color_func)
                    print(f"已应用颜色图片的着色（从图片中随机提取 {len(colors)} 种颜色）。")
                else:
                    print("颜色提取失败，使用默认颜色。")
            except Exception as e:
                print(f"加载颜色图片失败: {e}")

        # 保存文件
        output_filename = os.path.join(OUTPUT_FOLDER, "wordcloud_result.png")
        wc.to_file(output_filename)
        print(f"词云已保存至: {output_filename}")

        # 展示
        plt.figure(figsize=(10, 10))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    # 检查 regex 库是否安装 (因为使用了 \p{P})
    try:
        import regex
    except ImportError:
        print("错误：请先安装 regex 库 (pip install regex)，标准 re 库不支持 \\p{P} 语法。")
        exit()

    generator = BiliDanmakuWordCloud(XML_FILE_PATH)
    generator.load_data()
    counts = generator.process_text()
    
    if counts:
        generator.generate_wordcloud(counts)
    else:
        print("没有提取到足够的词汇，无法生成词云。")