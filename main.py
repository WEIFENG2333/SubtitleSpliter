import os
import re
from ASRData import ASRData, from_srt, ASRDataSeg

import difflib
from typing import List
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from split_by_llm import split_by_llm

MAX_WORD_COUNT = 16  # Maximum words for English or characters for Chinese


def merge_segments_based_on_sentences(asr_data: ASRData, sentences: List[str]):
    # 将 asr_data.segments 中的每个字符映射到其对应的段索引
    asr_text = ''.join(seg.text for seg in asr_data.segments)
    asr_char_to_seg_index = []
    for idx, seg in enumerate(asr_data.segments):
        asr_char_to_seg_index.extend([idx] * len(seg.text))
    
    new_segments = []
    asr_index = 0  # 记录在 asr_text 中的位置
    asr_len = len(asr_text)
    threshold = 0.5  # 相似度阈值，可根据需要调整
    max_shift = 30   # 滑动窗口的最大偏移量

    for sentence in sentences:
        # print(f"[+]正在匹配句子: {sentence}")
        sentence_len = len(sentence)
        best_ratio = 0.0
        best_pos = None

        # 滑动窗口的范围，从 asr_index 开始，窗口大小从最大到最小
        max_window_size = min(sentence_len * 4, asr_len - asr_index)
        min_window_size = max(1, sentence_len // 2)  # 最小窗口大小，防止窗口过小
        # 生成以 sentence_len 为中心的窗口大小列表
        window_sizes = list(range(min_window_size, max_window_size + 1))
        window_sizes.sort(key=lambda x: abs(x - sentence_len))

        for window_size in window_sizes:
            # print(f"window_size: {window_size}")
            # 滑动窗口的起始位置范围，限制在 max_shift 以内
            for start in range(asr_index, min(asr_index + max_shift + 1, asr_len - window_size + 1)):
                substr = asr_text[start:start + window_size]
                ratio = difflib.SequenceMatcher(None, sentence, substr).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_pos = start
                    best_window_size = window_size
                if ratio == 1.0:
                    break  # 完全匹配，跳出循环
            if best_ratio == 1.0:
                break  # 完全匹配，跳出循环

        if best_ratio >= threshold and best_pos is not None:
            start_char_pos = best_pos
            end_char_pos = best_pos + best_window_size - 1

            # 获取对应的段索引
            start_seg_index = asr_char_to_seg_index[start_char_pos]
            end_seg_index = asr_char_to_seg_index[end_char_pos]

            # 合并从 start_seg_index 到 end_seg_index 的段
            merged_text = ''.join(seg.text for seg in asr_data.segments[start_seg_index:end_seg_index+1])
            merged_start_time = asr_data.segments[start_seg_index].start_time
            merged_end_time = asr_data.segments[end_seg_index].end_time
            # print(f"[+]匹配到的文本: {merged_text}")
            merged_seg = ASRDataSeg(merged_text, merged_start_time, merged_end_time)

            # 如果合并后的文本长度超过16，则进行拆分
            if len(merged_text) > 16:
                # 获取合并的所有段
                segs_to_merge = asr_data.segments[start_seg_index:end_seg_index+1]
                # 调用拆分函数
                split_segs = split_long_segment(merged_text, segs_to_merge)
                new_segments.extend(split_segs)
            else:
                new_segments.append(merged_seg)

            asr_index = best_pos + best_window_size  # 更新 asr_index 到当前匹配的结束位置
        else:
            # 未找到匹配，处理方式取决于需求，这里选择跳过或直接添加句子
            print(f"[-]未能匹配的句子: {sentence}")
            print(''.join(seg.text for seg in asr_data.segments[asr_index:asr_index+20]))
            pass

    # 更新 asr_data.segments
    return ASRData(new_segments)

def split_long_segment(merged_text: str, segs_to_merge: List[ASRDataSeg]) -> List[ASRDataSeg]:
    """
    如果 merged_text 长度超过16，按照最大时间间隔进行拆分。
    在寻找时间间隔最大的两个相邻段之间的位置时，只在中间2/3长度的区间内寻找。
    :param merged_text: 合并后的文本
    :param segs_to_merge: 合并的原始段列表
    :return: 拆分后的 ASRDataSeg 列表
    """
    # 初始化结果列表
    result_segs = []

    # 如果文本长度小于等于16，直接返回
    if len(merged_text) <= 16 or len(segs_to_merge) == 1:
        merged_seg = ASRDataSeg(merged_text, segs_to_merge[0].start_time, segs_to_merge[-1].end_time)
        result_segs.append(merged_seg)
        return result_segs

    # 找到时间间隔最大的两个相邻段之间的位置，只在中间2/3长度的区间寻找
    n = len(segs_to_merge)
    start_idx = n // 6  # 从1/6位置开始
    end_idx = (5 * n) // 6  # 到5/6位置结束

    max_gap = 0
    split_index = None  # 在 segs_to_merge 中的分割索引

    for i in range(start_idx, end_idx):
        gap = segs_to_merge[i + 1].start_time - segs_to_merge[i].end_time
        if gap > max_gap:
            max_gap = gap
            split_index = i

    if split_index is None:
        # 如果没有找到有效的分割点，或者所有段时间连续
        # 那么就在中间位置进行拆分
        split_index = n // 2
        print(f"[-]未能找到有效的分割点，在中间位置进行拆分: {merged_text}")

    # 拆分段
    first_segs = segs_to_merge[:split_index + 1]
    second_segs = segs_to_merge[split_index + 1:]

    # 获取拆分后的文本
    first_text = ''.join(seg.text for seg in first_segs)
    second_text = ''.join(seg.text for seg in second_segs)

    # 递归处理拆分后的段
    result_segs.extend(split_long_segment(first_text, first_segs))
    result_segs.extend(split_long_segment(second_text, second_segs))

    return result_segs


if __name__ == '__main__':
    asr_data = from_srt(open("test_data/yidali.srt", encoding="utf-8").read())
    # 优化：合并两个操作，减少循环次数
    asr_data.segments = []
    for seg in asr_data.segments:
        # 删除纯标点符号的段
        if re.sub(r'[^\w]', '', seg.text):
            # 处理英文单词
            if re.match(r'^[a-zA-Z]+$', seg.text):
                seg.text = seg.text.lower() + " "
            asr_data.segments.append(seg)

    txt = asr_data.to_txt().replace("\n", "")
    print(txt)

    print("[+]正在请求LLM进行断句...")
    sentences = split_by_llm(txt)
    
    new_asr_data = merge_segments_based_on_sentences(asr_data, sentences)
    # print(len(new_asr_data.segments))

    # 查看合并后的 segments
    new_asr_data.to_srt(save_path="test_data/yidali_merged.srt")



