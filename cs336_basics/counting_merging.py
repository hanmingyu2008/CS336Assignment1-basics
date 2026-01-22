from pretokenization import parallel_pre_tokenize_file

def prepare_initial_tokens_no_positions(token_counts, special_tokens):
    """将频率统计转换为 tokens 列表"""
    tokens = []
    
    # 特殊 tokens（作为整体）
    for special in special_tokens:
        special_bytes = special.encode('utf-8')
        count = token_counts.get(special_bytes, 0)
        # 存储为 [bytes_object] 而不是 [int, int, ...]
        tokens.append(([special_bytes], count))
    
    # 普通 tokens（拆分为 bytes）
    for token_bytes, count in token_counts.items():
        # 跳过已经是 special token 的
        try:
            if token_bytes.decode('utf-8') in special_tokens:
                continue
        except:
            pass
        
        # 拆分为整数列表
        byte_list = list(token_bytes)  # b"hello" -> [104, 101, 108, 108, 111]
        tokens.append((byte_list, count))
    
    return tokens  # 现在每个 token 是 (byte_list, count)

def initial_pair_computing(initial_tokens):
    # 统计所有相邻对的出现次数和位置
    pair_counts = {}      # {(byte1, byte2): total_frequency}
    pair_positions = {}   # {(byte1, byte2): [(token_idx, position, weight), ...]}

    for token_idx, (byte_list, count) in enumerate(initial_tokens):
        for pos in range(len(byte_list) - 1):
            byte1 = byte_list[pos]
            byte2 = byte_list[pos + 1]
            pair = (byte1, byte2)
            
            # 累加频率（考虑权重）
            pair_counts[pair] = pair_counts.get(pair, 0) + count
            
            # 记录位置
            if pair not in pair_positions:
                pair_positions[pair] = []
            pair_positions[pair].append((token_idx, pos, count))
    return pair_counts,pair_positions

def merge_and_update(best_pair, new_token_id, initial_tokens, pair_counts, pair_positions):
    """
    合并字节对并更新所有相关数据结构
    """
    byte1, byte2 = best_pair
    
    # 获取所有出现位置
    positions = pair_positions.get(best_pair, [])
    for token_idx, pos, weight in positions:
        byte_list, count = initial_tokens[token_idx]
        
        # 跳过无效位置
        if pos >= len(byte_list) - 1:
            continue
        if byte_list[pos] != byte1 or byte_list[pos + 1] != byte2:
            continue
        
        # === 1. 删除受影响的旧字节对 ===
        
        # 左边字节对 (left, A) 减少
        if pos > 0:
            left_byte = byte_list[pos - 1]
            left_pair = (left_byte, byte1)
            pair_counts[left_pair] -= weight
            # 也要从 pair_positions 中移除对应条目
        
        # 被合并的字节对 (A, B) 完全移除
        pair_counts[best_pair] -= weight
        
        # 右边字节对 (B, right) 减少
        if pos < len(byte_list) - 2:
            right_byte = byte_list[pos + 2]
            right_pair = (byte2, right_byte)
            pair_counts[right_pair] -= weight
        
        # === 2. 执行合并 ===
        # 用 new_token_id 替换 [A, B]
        byte_list[pos] = new_token_id  # 新的合并 token
        byte_list.pop(pos + 1)  # 移除 B
        
        # === 3. 添加新的字节对 ===
        
        # 新的左边 (left, AB)
        if pos > 0:
            left_byte = byte_list[pos - 1]
            new_left_pair = (left_byte, new_token_id)
            pair_counts[new_left_pair] = pair_counts.get(new_left_pair, 0) + weight
            if new_left_pair not in pair_positions:
                pair_positions[new_left_pair] = []
            pair_positions[new_left_pair].append((token_idx, pos-1, count))

        # 新的右边 (AB, right)
        if pos < len(byte_list) - 1:  # 注意：现在 AB 在位置 pos
            right_byte = byte_list[pos + 1]
            new_right_pair = (new_token_id, right_byte)
            pair_counts[new_right_pair] = pair_counts.get(new_right_pair, 0) + weight
            if new_right_pair not in pair_positions:
                pair_positions[new_right_pair] = []
            pair_positions[new_right_pair].append((token_idx, pos, count))
        
        # 更新 token 数据
        initial_tokens[token_idx] = (byte_list, count)
    
    # 清理频率为0的字节对
    pair_counts = {k: v for k, v in pair_counts.items() if v > 0}
    
    return initial_tokens, pair_counts, pair_positions

def merging(initial_tokens_counts, vocab_size, special_tokens):
    
    vocab = {}
    
    # 添加基础字节
    for i in range(256):
        vocab[i] = bytes([i])
    for i,spe in enumerate(special_tokens):
        vocab[256+i] = spe    
    
    merges = []

    initial_tokens = prepare_initial_tokens_no_positions(initial_tokens_counts, special_tokens)
    
    
    pair_counts, pair_positions = initial_pair_computing(initial_tokens)
    
     
    with open("vocab.txt","w") as filee:
        
        while len(vocab) < vocab_size:
            print({"".join([vocab[x].decode() for x in a]): pair_counts[a] for a in pair_counts},file=filee)
            best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
            
            a,b = best_pair
            new_token_id = len(vocab)
            merges.append((vocab[a],vocab[b]))
            vocab[new_token_id] = vocab[a]+vocab[b]

            initial_tokens,pair_counts,pair_positions = merge_and_update(best_pair, new_token_id,initial_tokens,pair_counts,pair_positions)
        print(vocab,file=filee)
        print(merges,file=filee)
    return vocab,merges

if __name__ == "__main__":
    initial_token_counts =  parallel_pre_tokenize_file("a.txt",["<|endoftext|>"])
    with open("vocab.txt","w") as filee:
        print(initial_token_counts, file=filee)
    vocab,merges = merging(initial_token_counts, 280, ["<|endoftext|>"])