from __future__ import annotations

import os
import regex as re
from collections import defaultdict
from typing import Dict, Tuple

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def to_bytes_tuple(word: str) -> Tuple[bytes]:
    l = list(word.encode("utf-8"))
    l = [bytes([x]) for x in l]
    return tuple(l)

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # Step 1: 构建词典:0~255字节+特殊字符
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256

    special_token_bytes = [token.encode("utf-8") for token in special_tokens]
    for token_bytes in special_token_bytes:
        if token_bytes not in vocab.values():
            vocab[next_id] = token_bytes
            next_id += 1

    # Step 2: 预词元化
    pre_tokens_cnt = defaultdict(int)

    def to_bytes_tuple(word: str) -> Tuple[bytes]:
        l = list(tuple(word.encode("utf-8")))
        l = [bytes([x]) for x in l]
        return tuple(l)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    chunks = re.split("|".join(map(re.escape, special_tokens)), text)
    
    for chunk in chunks:
        for m in re.finditer(PAT, chunk):
            word = m.group(0)
            pre_tokens_cnt[to_bytes_tuple(word)] += 1

    # Step 3: 重复合并过程并且记录
    merges = []

    while len(vocab) < vocab_size:
        pair_counts = defaultdict(int)

        for token, cnt in pre_tokens_cnt.items():
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                pair_counts[pair] += cnt

        if not pair_counts:
            break  

        max_count = max(pair_counts.values())
        candidates = [k for k, v in pair_counts.items() if v == max_count]
        best_pair = max(candidates)

        a, b = best_pair

        # 形成新的词元
        new_token = a + b
        vocab[next_id] = new_token
        next_id += 1

        changes = []
        for token, cnt in pre_tokens_cnt.items():
            indices = [i for i in range(len(token) - 1) if token[i:i + 2] == best_pair]
            if indices:
                new_pre_token = []
                i = 0
                while i < len(token):
                    if i in indices:
                        new_pre_token.append(new_token)
                        i += 2
                    else:
                        new_pre_token.append(token[i])
                        i += 1
                new_pre_token = tuple(new_pre_token)
                changes.append((token, new_pre_token, cnt))

        # 应用变更
        for old_token, new_pre_token, cnt in changes:
            pre_tokens_cnt[new_pre_token] = pre_tokens_cnt.get(new_pre_token, 0) + cnt
            del pre_tokens_cnt[old_token]

        merges.append((a, b))

    return vocab, merges