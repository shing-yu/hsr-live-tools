def pm9(input_str) -> str:
    """
    将20位以下的数字字符串摘要为9位数字。
    """
    input_str = str(input_str)
    # 1. 校验输入（可选）
    if not input_str.isdigit():
        raise ValueError("输入必须仅包含数字")
    
    # 2. 定义常量 (大素数用于混合)
    P1 = 998244353
    P2 = 1000000007
    MOD = 1000000000  # 保证输出为9位
    
    # 3. 初始化 Hash 值 (你可以修改这个种子值当作密钥)
    h_val = 314159265
    
    # 4. 混合处理
    for char in input_str:
        d = int(char)
        
        # 步骤 A: 乘法扩散
        h_val = (h_val * P1) + d
        
        # 步骤 B: 位运算非线性混合 (异或 + 移位)
        # 这里的 & 0xFFFFFFFFFFFFFFFF 是为了在Python中模拟64位整数溢出，
        # 防止数字无限变大降低效率，但在逻辑上不是必须的。
        h_val = h_val ^ (h_val >> 5)
        
        # 步骤 C: 二次混合
        h_val = (h_val * P2) + (d << 3)
    
    # 5. 最终混淆 (Avalanche effect)
    h_val = h_val ^ (h_val >> 17)
    
    # 6. 取模得到9位结果
    result = h_val % MOD
    
    # 7. 格式化为字符串，不足9位补零
    return f"PM{result:09d}"