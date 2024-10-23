import os
import re
from ASRData import ASRData, from_srt, ASRDataSeg

import difflib
from typing import List
import sys

from split_by_llm import split_by_llm

MAX_WORD_COUNT = 16  # 英文单词或中文字符的最大数量


def is_pure_punctuation(s: str) -> bool:
    """
    检查字符串是否仅由标点符号组成
    """
    return not re.search(r'\w', s, flags=re.UNICODE)


def count_words(text: str) -> int:
    """
    统计混合文本中英文单词数和中文字符数的总和
    """
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_text = re.sub(r'[\u4e00-\u9fff]', ' ', text)
    english_words = len(english_text.strip().split())
    return english_words + chinese_chars


def preprocess_text(s: str) -> str:
    """
    通过转换为小写并规范化空格来标准化文本
    """
    return ' '.join(s.lower().split())


def merge_segments_based_on_sentences(asr_data: ASRData, sentences: List[str]) -> ASRData:
    """
    基于提供的句子列表合并ASR分段
    """
    asr_texts = [seg.text for seg in asr_data.segments]
    asr_len = len(asr_texts)
    asr_index = 0  # 当前分段索引位置
    threshold = 0.5  # 相似度阈值
    max_shift = 10   # 滑动窗口的最大偏移量

    new_segments = []

    for sentence in sentences:
        print(f"[+] 处理句子: {sentence}")
        sentence_proc = preprocess_text(sentence)
        word_count = count_words(sentence_proc)
        best_ratio = 0.0
        best_pos = None
        best_window_size = 0

        # 滑动窗口大小，优先考虑接近句子词数的窗口
        max_window_size = min(word_count * 2, asr_len - asr_index)
        min_window_size = max(1, word_count // 2)

        window_sizes = sorted(range(min_window_size, max_window_size + 1), key=lambda x: abs(x - word_count))

        for window_size in window_sizes:
            max_start = min(asr_index + max_shift + 1, asr_len - window_size + 1)
            for start in range(asr_index, max_start):
                substr = ''.join(asr_texts[start:start + window_size])
                substr_proc = preprocess_text(substr)
                ratio = difflib.SequenceMatcher(None, sentence_proc, substr_proc).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_pos = start
                    best_window_size = window_size
                if ratio == 1.0:
                    break  # 完全匹配
            if best_ratio == 1.0:
                break  # 完全匹配

        if best_ratio >= threshold and best_pos is not None:
            start_seg_index = best_pos
            end_seg_index = best_pos + best_window_size - 1

            # 合并分段
            merged_text = ''.join(asr_texts[start_seg_index:end_seg_index + 1])
            merged_start_time = asr_data.segments[start_seg_index].start_time
            merged_end_time = asr_data.segments[end_seg_index].end_time
            merged_seg = ASRDataSeg(merged_text, merged_start_time, merged_end_time)

            print(f"[+] 合并分段: {merged_seg.text}")
            print("=============")

            # 拆分超过最大词数的分段
            if count_words(merged_text) > MAX_WORD_COUNT:
                segs_to_merge = asr_data.segments[start_seg_index:end_seg_index + 1]
                split_segs = split_long_segment(merged_text, segs_to_merge)
                new_segments.extend(split_segs)
            else:
                new_segments.append(merged_seg)

            asr_index = end_seg_index + 1  # 移动到下一个未处理的分段
        else:
            # 无法匹配句子，跳过当前分段
            print(f"[-] 无法匹配句子: {sentence}")
            asr_index += 1

    return ASRData(new_segments)


def split_long_segment(merged_text: str, segs_to_merge: List[ASRDataSeg]) -> List[ASRDataSeg]:
    """
    基于最大时间间隔拆分长分段，尽可能避免拆分英文单词
    """
    result_segs = []
    print(f"[+] 正在拆分长分段: {merged_text}")

    # 基本情况：如果分段足够短或无法进一步拆分
    if count_words(merged_text) <= MAX_WORD_COUNT or len(segs_to_merge) == 1:
        merged_seg = ASRDataSeg(
            merged_text.strip(),
            segs_to_merge[0].start_time,
            segs_to_merge[-1].end_time
        )
        result_segs.append(merged_seg)
        return result_segs


    # 在分段中间2/3部分寻找最佳拆分点
    n = len(segs_to_merge)
    start_idx = n // 6
    end_idx = (5 * n) // 6

    split_index = max(
        range(start_idx, end_idx),
        key=lambda i: segs_to_merge[i + 1].start_time - segs_to_merge[i].end_time,
        default=None
    )

    if split_index is None:
        split_index = n // 2

    first_segs = segs_to_merge[:split_index + 1]
    second_segs = segs_to_merge[split_index + 1:]

    first_text = ''.join(seg.text for seg in first_segs)
    second_text = ''.join(seg.text for seg in second_segs)

    # 必要时递归拆分
    result_segs.extend(split_long_segment(first_text, first_segs))
    result_segs.extend(split_long_segment(second_text, second_segs))

    return result_segs


if __name__ == '__main__':
    # 从SRT文件加载ASR数据
    srt_path = "test_data/yidali.srt"
    with open(srt_path, encoding="utf-8") as f:
        asr_data = from_srt(f.read())

    # 处理英文（小写且加空格）和标点符号（删除）
    new_segments = []
    for seg in asr_data.segments:
        if not is_pure_punctuation(seg.text):
            if re.match(r'^[a-zA-Z\']+$', seg.text.strip()):
                seg.text = seg.text.lower() + " "
            new_segments.append(seg)
    asr_data.segments = new_segments

    # 获取连接后的文本
    txt = asr_data.to_txt().replace("\n", "")
    print(txt)

    # 使用LLM将文本拆分为句子
    print("[+] 正在请求LLM将文本拆分为句子...")
    sentences = split_by_llm(txt, use_cache=True)

    # 基于LLM已经分段的句子，对ASR分段进行合并
    new_asr_data = merge_segments_based_on_sentences(asr_data, sentences)

    # 保存到SRT文件
    new_srt_path = srt_path.replace(".srt", "_merged.srt")
    new_asr_data.to_srt(save_path=new_srt_path)

