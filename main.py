import xml.etree.ElementTree as ET
import csv
import regex as re
import os
import time
from collections import defaultdict, Counter
from datetime import datetime

from snownlp import SnowNLP
from snownlp import sentiment

# =================配置区域=================
INVALID_REGEX_PATTERNS = [
    r'^.*?加强.*?$',
    r'开.*?门|门.*?开',
    r'^(\[[^\]]*\])+$',
    r'^\s*$',
    r'^.*?表情【.*?$',
    r'记忆是梦的开场白',
    r'与.*共舞',
    r'^[\p{P}]+$',
    r'^6+$',
    r'^[a-zA-Z0-9]+$',
    r'^.*?来了.*?$'
]

# 模型路径
MODEL_PATH = './snow/hsr3.8.marshal' 
sentiment.load(MODEL_PATH)

# ==========================================

class BilibiliLiveAnalyzer:
    def __init__(self, xml_file_path, output_folder='output'):
        self.xml_file_path = xml_file_path
        self.output_folder = output_folder
        self.danmakus = [] 
        self.gifts = []    
        self.regex_list = [re.compile(p) for p in INVALID_REGEX_PATTERNS]
        
        self.effective_danmakus = [] 
        self.user_stats = defaultdict(lambda: {'name': '', 'msgs': []}) 
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)


    def load_and_parse(self):
        """模块1：数据加载与解析"""
        print(f"正在加载文件: {self.xml_file_path} ...")
        try:
            tree = ET.parse(self.xml_file_path)
            root = tree.getroot()
            
            for child in root.findall('d'):
                attr = child.attrib
                p_attr = attr.get('p', '').split(',')
                uid = attr.get('uid') or (p_attr[6] if len(p_attr) > 6 else '0')
                ts = attr.get('timestamp') or (p_attr[4] if len(p_attr) > 4 else '0')
                
                self.danmakus.append({
                    'text': str(child.text).strip(),
                    'uid': uid,
                    'user': attr.get('user', '未知用户'),
                    'timestamp': float(ts),
                    'is_gift': False
                })

            for child in root.findall('s'):
                attr = child.attrib
                self.gifts.append({
                    'text': child.text if child.text else '', 
                    'uid': attr.get('uid'),
                    'user': attr.get('username'),
                    'price': float(attr.get('price', 0)),
                    'num': int(attr.get('num', 1)),
                    'giftname': attr.get('giftname'),
                    'timestamp': float(attr.get('timestamp', 0)),
                    'is_gift': True
                })
                
            print(f"解析完成。弹幕数: {len(self.danmakus)}, 礼物记录数: {len(self.gifts)}")
            
        except Exception as e:
            print(f"解析XML出错: {e}")

    def _is_effective(self, text):
        if not text: return False
        for pattern in self.regex_list:
            if pattern.search(text):
                return False
        return True

    def process_data(self):
        """模块2：数据预处理（适配嵌套结构 + 宽松逻辑）"""
        print("正在筛选有效弹幕...")
        
        pending_process = []
        for d in self.danmakus:
            self.user_stats[d['uid']]['name'] = d['user']
            self.user_stats[d['uid']]['msgs'].append(d['text'])
            if self._is_effective(d['text']):
                pending_process.append(d)
        
        print(f"有效弹幕筛选完毕，共 {len(pending_process)} 条，准备进行情感分析...")

        # 1. 批量推理
        start_time = time.time()
        
        for i, d in enumerate(pending_process):
            # 打印进度
            if (i + 1) % 1000 == 0:
                print(f"已处理 {i + 1}/{len(pending_process)} 条弹幕...")
            
            text = d['text']
            
            # 使用SnowNLP进行情感分析
            s = SnowNLP(text)
            # s.sentiments 的返回值是一个介于0和1之间的浮点数，越接近1越积极
            sentiment_score = s.sentiments

            # 根据分数分类
            if sentiment_score > 0.65:
                sentiment_type = 'positive'
            elif sentiment_score < 0.35:
                sentiment_type = 'negative'
            else:
                sentiment_type = 'neutral'
        
            # 更新数据
            d_processed = d.copy()
            d_processed.update({
                'sentiment_score': sentiment_score,
                'sentiment_type': sentiment_type,
                'raw_label': sentiment_type
            })
            self.effective_danmakus.append(d_processed)
        
        end_time = time.time()
        print(f"情感分析完成，耗时: {end_time - start_time:.2f}秒")
        print(f"处理完成，有效弹幕库已生成。")

    def _format_freq_list(self, msg_list):
        c = Counter(msg_list)
        items = [f"{k}({v}次)" for k, v in c.most_common()]
        return " | ".join(items)

    def write_csv(self, filename, headers, rows):
        path = os.path.join(self.output_folder, filename)
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        print(f"已生成: {path}")

    # ================= 统计模块 =================

    def stat_overview(self):
        total_d = len(self.danmakus)
        danmaku_senders_uids = set(d['uid'] for d in self.danmakus)
        total_d_senders = len(danmaku_senders_uids)
        
        gift_value_no_sc = 0.0
        gift_senders_uids = set()
        sc_value = 0.0
        sc_count = 0
        
        for g in self.gifts:
            val = g['price'] * g['num']
            if g['giftname'] == '醒目留言':
                sc_value += val
                sc_count += 1
            else:
                gift_value_no_sc += val
                gift_senders_uids.add(g['uid'])
        
        rows = [[total_d, total_d_senders, round(gift_value_no_sc, 2), len(gift_senders_uids), round(sc_value, 2), sc_count]]
        self.write_csv('1_总体统计.csv', ['总弹幕数', '弹幕发送人数', '礼物总价值(无SC)', '送礼人数', 'SC总价值', 'SC总数量'], rows)

    def stat_top_danmaku_users(self):
        sorted_users = sorted(self.user_stats.items(), key=lambda x: len(x[1]['msgs']), reverse=True)[:20]
        rows = [[uid, data['name'], len(data['msgs']), self._format_freq_list(data['msgs'])] for uid, data in sorted_users]
        self.write_csv('2_弹幕发送Top20.csv', ['UID', '用户名', '发送数量', '弹幕频率列表'], rows)

    def stat_top_gift_users(self):
        gift_users = defaultdict(lambda: {'name': '', 'total_value': 0, 'gifts': []})
        for g in self.gifts:
            val = g['price'] * g['num']
            gift_users[g['uid']]['name'] = g['user']
            gift_users[g['uid']]['total_value'] += val
            gift_users[g['uid']]['gifts'].append(f"{g['giftname']}x{g['num']}")
        
        sorted_gifts = sorted(gift_users.items(), key=lambda x: x[1]['total_value'], reverse=True)[:20]
        rows = [[uid, data['name'], round(data['total_value'], 2), self._format_freq_list(data['gifts'])] for uid, data in sorted_gifts]
        self.write_csv('3_礼物贡献Top20.csv', ['UID', '用户名', '礼物总价值', '礼物列表'], rows)

    def stat_all_sc(self):
        sc_list = sorted([g for g in self.gifts if g['giftname'] == '醒目留言'], key=lambda x: x['price'], reverse=True)
        rows = [[sc['uid'], sc['user'], sc['price'], sc['text']] for sc in sc_list]
        self.write_csv('4_所有SC记录.csv', ['UID', '用户名', '价值', '留言内容'], rows)

    def stat_effective_count(self):
        self.write_csv('5_有效弹幕统计.csv', ['有效弹幕数量'], [[len(self.effective_danmakus)]])

    def stat_sentiment_overview(self):
        if not self.effective_danmakus: return
        scores = [d['sentiment_score'] for d in self.effective_danmakus]
        types = [d['sentiment_type'] for d in self.effective_danmakus]
        c = Counter(types)
        avg_score = sum(scores) / len(scores)
        rows = [[c['positive'], c['negative'], c['neutral'], round(avg_score, 4)]]
        self.write_csv('6_情感分析总览.csv', ['积极数量', '消极数量', '中性数量', '平均情感置信度'], rows)

    def stat_sentiment_users(self):
        neg_users = defaultdict(int)
        neu_users = defaultdict(int)
        for d in self.effective_danmakus:
            if d['sentiment_type'] == 'negative': neg_users[d['uid']] += 1
            elif d['sentiment_type'] == 'neutral': neu_users[d['uid']] += 1
        
        rows = []
        for uid, count in sorted(neg_users.items(), key=lambda x: x[1], reverse=True)[:5]:
            rows.append(['消极Top5', uid, self.user_stats[uid]['name'], count, self._format_freq_list(self.user_stats[uid]['msgs'])])
        for uid, count in sorted(neu_users.items(), key=lambda x: x[1], reverse=True)[:3]:
            rows.append(['中性Top3', uid, self.user_stats[uid]['name'], count, self._format_freq_list(self.user_stats[uid]['msgs'])])
        self.write_csv('7_特定情感倾向用户Top.csv', ['榜单类型', 'UID', '用户名', '特定情感弹幕数', '所有弹幕列表'], rows)

    def stat_time_trend(self):
        # 请根据你的视频实际开始时间修改这里！
        START_TS = 1764932400 
        STEP = 30
        
        timestamps = [d['timestamp'] for d in self.danmakus] + [g['timestamp'] for g in self.gifts]
        if not timestamps: timestamps = [START_TS]
        end_ts = max(timestamps)
        
        data_buckets = defaultdict(lambda: {'d_count': 0, 'd_pos': 0, 'd_neg': 0, 'd_neu': 0, 'g_val': 0.0})

        def get_bucket_index(ts):
            if ts < START_TS: return -1
            return int((ts - START_TS) / STEP)

        for d in self.danmakus:
            idx = get_bucket_index(d['timestamp'])
            if idx >= 0: data_buckets[idx]['d_count'] += 1
        
        for d in self.effective_danmakus:
            idx = get_bucket_index(d['timestamp'])
            if idx >= 0:
                if d['sentiment_type'] == 'positive':
                    data_buckets[idx]['d_pos'] += 1
                elif d['sentiment_type'] == 'negative':
                    data_buckets[idx]['d_neg'] += 1
                else:
                    data_buckets[idx]['d_neu'] += 1

        for g in self.gifts:
            idx = get_bucket_index(g['timestamp'])
            if idx >= 0:
                data_buckets[idx]['g_val'] += (g['price'] * g['num'])

        rows = []
        start_dt_str = datetime.fromtimestamp(START_TS).strftime('%H:%M:%S')
        rows.append([start_dt_str, 0, 0, 0, 0, 0, 0.0, 0.0])

        total_intervals = int((end_ts - START_TS) / STEP) + 1
        acc_d_count = 0
        acc_g_val = 0.0
        
        for i in range(total_intervals + 1):
            curr = data_buckets[i]
            acc_d_count += curr['d_count']
            acc_g_val += curr['g_val']
            
            end_of_interval_ts = START_TS + ((i + 1) * STEP) 
            dt_str = datetime.fromtimestamp(end_of_interval_ts).strftime('%H:%M:%S')
            
            rows.append([
                dt_str, 
                curr['d_count'],            
                curr['d_pos'],              
                curr['d_neu'],              
                curr['d_neg'],              
                acc_d_count,                
                round(curr['g_val'] / 100, 2), 
                round(acc_g_val / 100, 2)      
            ])

        headers = ['时间轴', '当前30s总弹幕数', '积极弹幕数', '中性弹幕数', '消极弹幕数', '累计弹幕数', '当前30s礼物(元)', '累计礼物(元)']
        self.write_csv('8_趋势统计_详细情感.csv', headers, rows)

    def export_debug_files(self):
        """新增任务：导出分类后的文本用于人工核查"""
        files = {
            'positive': 'debug_positive.txt',   # 积极
            'negative': 'debug_negative.txt',   # 消极
            'neutral':  'debug_neutral.txt'     # 中性
        }
        
        # 准备容器
        data_map = defaultdict(list)
        
        for d in self.effective_danmakus:
            stype = d['sentiment_type']
            
            # 格式设计: [置信度] 文本
            # 例如: [0.9982] 这种垃圾游戏赶紧倒闭吧
            # 保留4位小数，方便你观察那些 0.59 vs 0.61 的临界值数据
            line = f"[{d['sentiment_score']:.4f}] {d['text']}"
            data_map[stype].append(line)
            
        # 分别写入三个文件
        for stype, filename in files.items():
            path = os.path.join(self.output_folder, filename)
            content_list = data_map[stype]
            
            # 可选：按分数排序，方便观察 (消极按分数低到高，积极按分数高到低)
            # if stype == 'positive':
            #     content_list.sort(key=lambda x: float(x[1:7]), reverse=True)
            # elif stype == 'negative':
            #     content_list.sort(key=lambda x: float(x[1:7]))

            with open(path, 'w', encoding='utf-8') as f:
                # 写入头部说明
                f.write(f"=== {stype} 类别 (共 {len(content_list)} 条) ===\n")
                f.write("格式: [模型置信度] 弹幕内容\n")
                f.write("说明: 0接近纯消极，1接近纯积极。中性通常是因为分数在0.4-0.6之间。\n")
                f.write("--------------------------------------------------\n")
                
                # 写入所有内容
                f.write('\n'.join(content_list))
                
            print(f"已生成人工核查文件: {path}")

    def run_all(self):
        self.load_and_parse()
        self.process_data()
        self.stat_overview()
        self.stat_top_danmaku_users()
        self.stat_top_gift_users()
        self.stat_all_sc()
        self.stat_effective_count()
        self.stat_sentiment_overview()
        self.stat_sentiment_users()
        self.stat_time_trend()
        self.export_debug_files()
        print("所有统计任务完成！")

if __name__ == "__main__":
    xml_path = 'input.xml' 
    tool = BilibiliLiveAnalyzer(xml_path)
    tool.run_all()