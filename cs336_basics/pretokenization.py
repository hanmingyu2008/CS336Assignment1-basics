import regex as re
from collections import defaultdict
from typing import List, Dict, Tuple
import multiprocessing
from functools import partial

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def split_by_special_tokens(text: str, special_tokens: List[str]) -> List[str]:
    """
    按 special tokens 切分文本
    
    Args:
        text: 输入文本
        special_tokens: 特殊 token 列表（如 ["<|endoftext|>"]）
    
    Returns:
        切分后的片段列表
    """
    if not special_tokens or not text:
        return [text]
    escaped_tokens = [re.escape(token) for token in special_tokens]
    
    pattern = "|".join(escaped_tokens)
    
    segments = re.split(f'({pattern})', text)
    return [seg for seg in segments if seg]


def pre_tokenize_segment(segment: str, is_special_token: bool = False) -> Dict[bytes, int]:
    """
    对单个文本片段进行预分词
    
    Args:
        segment: 文本片段
        is_special_token: 是否为 special token
    
    Returns:
        该片段中 token 的频率统计 {token_bytes: count}
    """
    token_counts = defaultdict(int)
    
    if is_special_token:
        # 如果是 special token，作为一个整体
        token_counts[segment.encode('utf-8')] = 1
    else:
        for match in re.finditer(PAT, segment):
            token = match.group(0)
            if token:
                token_bytes = token.encode('utf-8')
                token_counts[token_bytes] += 1
    
    return dict(token_counts)

def pre_tokenize_text(text: str, special_tokens: List[str] = None) -> Dict[bytes, int]:
    """
    对完整文本进行预分词并统计频率
    
    Args:
        text: 输入文本
        special_tokens: 特殊 token 列表
    
    Returns:
        所有 token 的频率统计 {token_bytes: count}
    """
    if special_tokens is None:
        special_tokens = []
    
    # 如果 text 是 bytes，转换为 str
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='ignore')
    
    # 按 special tokens 切分文本
    segments = split_by_special_tokens(text, special_tokens)
    
    # 合并所有片段的统计结果
    final_counts = defaultdict(int)
    
    for segment in segments:
        is_special = segment in special_tokens
        segment_counts = pre_tokenize_segment(segment, is_special)
        
        for token_bytes, count in segment_counts.items():
            final_counts[token_bytes] += count
    
    return dict(final_counts)


def pre_tokenize_chunk(args: Tuple[int, int, str], special_tokens: List[str]) -> Dict[bytes, int]:
    """
    处理单个数据块（用于并行化）
    
    Args:
        args: (start_byte, end_byte, filepath)
        special_tokens: 特殊 token 列表
    
    Returns:
        该数据块的 token 频率统计
    """
    start, end, filepath = args
    
    try:
        with open(filepath, 'rb') as f:
            f.seek(start)
            chunk_bytes = f.read(end - start)
            
            # 尝试解码为 UTF-8
            try:
                chunk_text = chunk_bytes.decode('utf-8')
            except UnicodeDecodeError:
                # 如果失败，使用错误忽略模式
                chunk_text = chunk_bytes.decode('utf-8', errors='ignore')
            
            return pre_tokenize_text(chunk_text, special_tokens)
            
    except Exception as e:
        print(f"Error processing chunk {start}-{end}: {e}")
        return {}


def parallel_pre_tokenize_file(
    filepath: str, 
    special_tokens: List[str], 
    num_processes: int = 4
) -> Dict[bytes, int]:
    """
    并行预分词大文件
    
    Args:
        filepath: 文件路径
        special_tokens: 特殊 token 列表
        num_processes: 进程数
    
    Returns:
        整个文件的 token 频率统计
    """
    # 找到数据块边界（使用作业提供的函数）
    from pretokenization_example import find_chunk_boundaries
    
    with open(filepath, 'rb') as f:
        # 假设默认的 special token 是 <|endoftext|>
        default_special = b"<|endoftext|>"
        
        # 如果有传入 special_tokens，使用第一个作为切分 token
        if special_tokens:
            split_token = special_tokens[0].encode('utf-8')
        else:
            split_token = default_special
        
        boundaries = find_chunk_boundaries(f, num_processes, split_token)
    
    # 准备每个进程的参数
    chunk_args = []
    for i in range(len(boundaries) - 1):
        chunk_args.append((boundaries[i], boundaries[i + 1], filepath))
    
    print(f"Processing {len(chunk_args)} chunks...")
    
    # 并行处理
    final_counts = defaultdict(int)
    
    if num_processes > 1 and len(chunk_args) > 1:
        # 使用多进程
        with multiprocessing.Pool(processes=num_processes) as pool:
            process_func = partial(pre_tokenize_chunk, special_tokens=special_tokens)
            results = pool.map(process_func, chunk_args)
    else:
        # 单进程（用于调试）
        results = []
        for args in chunk_args:
            results.append(pre_tokenize_chunk(args, special_tokens))
    
    # 合并所有结果
    for result in results:
        for token_bytes, count in result.items():
            final_counts[token_bytes] += count
    
    print(f"Found {len(final_counts)} unique tokens")
    
    return dict(final_counts)


def prepare_initial_tokens(
    token_counts: Dict[bytes, int], 
    special_tokens: List[str]
) -> List[Tuple[List[bytes], int]]:
    """
    准备用于 BPE 训练的初始 tokens
    
    Args:
        token_counts: token 频率统计
        special_tokens: 特殊 token 列表
    
    Returns:
        列表，每个元素是 (token_bytes_list, count)
        special tokens 作为一个整体，普通 tokens 拆分为单个 bytes
    """
    tokens_for_bpe = []
    
    # 1. 先处理 special tokens（确保它们在前面）
    special_token_set = set(special_tokens)
    
    for special_token in special_tokens:
        special_bytes = special_token.encode('utf-8')
        count = token_counts.get(special_bytes, 0)
        # special token 作为一个整体 [b'<|endoftext|>']
        tokens_for_bpe.append(([special_bytes], count))
    
    # 2. 处理普通 tokens（排除已经是 special token 的）
    for token_bytes, count in token_counts.items():
        # 尝试解码以检查是否为 special token
        try:
            token_str = token_bytes.decode('utf-8')
            if token_str in special_token_set:
                continue  # 已经处理过了
        except:
            pass  # 如果解码失败，继续处理
        
        # 普通 token：拆分为单个 bytes
        byte_list = list(token_bytes)
        tokens_for_bpe.append((byte_list, count))
    
    return tokens_for_bpe


# ============ 测试函数 ============

def prepare_initial_tokens(
    token_counts: Dict[bytes, int], 
    special_tokens: List[str]
) -> Dict:
    """
    准备用于 BPE 训练的初始数据
    
    Args:
        token_counts: token 频率统计
        special_tokens: 特殊 token 列表
    
    Returns:
        {
            'special_tokens': [(bytes_object, count)],  # 整体 tokens
            'regular_tokens': [(byte_int_list, count)]  # 拆分为 bytes 的 tokens
        }
    """
    result = {
        'special_tokens': [],  # 存储 (bytes_object, count)
        'regular_tokens': []   # 存储 ([int_byte1, int_byte2, ...], count)
    }
    
    special_token_set = set(special_tokens)
    
    for token_bytes, count in token_counts.items():
        # 检查是否为 special token
        try:
            token_str = token_bytes.decode('utf-8')
            if token_str in special_token_set:
                result['special_tokens'].append((token_bytes, count))
                continue
        except:
            pass
        
        # 普通 token：拆分为单个 bytes（整数）
        byte_ints = list(token_bytes)  # bytes -> [int, int, ...]
        result['regular_tokens'].append((byte_ints, count))
    
    # 确保所有 special tokens 都出现（即使频率为0）
    for special_token in special_tokens:
        special_bytes = special_token.encode('utf-8')
        found = False
        for existing_bytes, _ in result['special_tokens']:
            if existing_bytes == special_bytes:
                found = True
                break
        if not found:
            result['special_tokens'].append((special_bytes, 0))
    
    return result

if __name__ == "__main__":
    
    result = parallel_pre_tokenize_file("../data/TinyStoriesV2-GPT4-valid.txt",["<|endoftext|>"])

    with open("a.txt", "w") as filee:
        print(result,file=filee)