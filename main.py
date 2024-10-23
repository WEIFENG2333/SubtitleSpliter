import os
import re
from ASRData import ASRData, from_srt, ASRDataSeg

import difflib
from typing import List
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from split_by_llm import split_by_llm

MAX_WORD_COUNT = 16  # 英文单词或中文字符的最大数量
SEGMENT_THRESHOLD = 1000  # 每个分段的最大字数
FIXED_NUM_THREADS = 4  # 固定的线程数量
SPLIT_RANGE = 30  # 在分割点前后寻找最大时间间隔的范围


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

    # 递归拆分
    result_segs.extend(split_long_segment(first_text, first_segs))
    result_segs.extend(split_long_segment(second_text, second_segs))

    return result_segs


def process_split_by_llm(asr_data_part: ASRData) -> List[str]:
    """
    调用 split_by_llm 处理单个 ASRData 分段，返回句子列表
    """
    txt = asr_data_part.to_txt().replace("\n", "")
    sentences = split_by_llm(txt, use_cache=True)
    return sentences


def split_asr_data(asr_data: ASRData, num_segments: int) -> List[ASRData]:
    """
    根据ASR分段中的时间间隔，将ASRData拆分成多个部分。
    处理步骤：
    1. 计算总字数，并确定每个分段的字数范围。
    2. 确定平均分割点。
    3. 在分割点前后一定范围内，寻找时间间隔最大的点作为实际的分割点。
    """
    total_segs = len(asr_data.segments)
    total_word_count = count_words(asr_data.to_txt())
    words_per_segment = total_word_count // num_segments
    split_indices = []

    if num_segments <= 1 or total_segs <= num_segments:
        return [asr_data]

    # 计算每个分段的大致字数 根据每段字数计算分割点
    split_indices = [i * words_per_segment for i in range(1, num_segments)]
    # 调整分割点：在每个平均分割点附近寻找时间间隔最大的点
    adjusted_split_indices = []
    for split_point in split_indices:
        # 定义搜索范围
        start = max(0, split_point - SPLIT_RANGE)
        end = min(total_segs - 1, split_point + SPLIT_RANGE)
        # 在范围内找到时间间隔最大的点
        max_gap = -1
        best_index = split_point
        for j in range(start, end):
            gap = asr_data.segments[j + 1].start_time - asr_data.segments[j].end_time
            if gap > max_gap:
                max_gap = gap
                best_index = j
        adjusted_split_indices.append(best_index)

    # 移除重复的分割点
    adjusted_split_indices = sorted(list(set(adjusted_split_indices)))

    # 根据调整后的分割点拆分ASRData
    segments = []
    prev_index = 0
    for index in adjusted_split_indices:
        part = ASRData(asr_data.segments[prev_index:index + 1])
        segments.append(part)
        prev_index = index + 1
    # 添加最后一部分
    if prev_index < total_segs:
        part = ASRData(asr_data.segments[prev_index:])
        segments.append(part)

    return segments


def determine_num_segments(word_count: int, threshold: int = 1000) -> int:
    """
    根据字数计算分段数，每1000个字为一个分段，至少为1
    """
    num_segments = word_count // threshold
    # 如果存在余数，增加一个分段
    if word_count % threshold > 0:
        num_segments += 1
    return max(1, num_segments)


def main(srt_path: str, save_path: str, num_threads: int = FIXED_NUM_THREADS):
    # 从SRT文件加载ASR数据
    with open(srt_path, encoding="utf-8") as f:
        asr_data = from_srt(f.read())

    # 预处理ASR数据，去除标点并转换为小写
    new_segments = []
    for seg in asr_data.segments:
        if not is_pure_punctuation(seg.text):
            if re.match(r'^[a-zA-Z\']+$', seg.text.strip()):
                seg.text = seg.text.lower() + " "
            new_segments.append(seg)
    asr_data.segments = new_segments

    # 获取连接后的文本
    txt = asr_data.to_txt().replace("\n", "")
    total_word_count = count_words(txt)
    print(f"[+] 合并后的文本长度: {total_word_count} 字")

    # 确定分段数
    num_segments = determine_num_segments(total_word_count, threshold=SEGMENT_THRESHOLD)
    print(f"[+] 根据字数 {total_word_count}，确定分段数: {num_segments}")

    # 分割ASRData
    asr_data_segments = split_asr_data(asr_data, num_segments)

    # for i in asr_data_segments:
    #     print(len(i.segments))
        # print(i.to_txt().split("\n"))

    # 多线程执行 split_by_llm 获取句子列表
    print("[+] 正在并行请求LLM将每个分段的文本拆分为句子...")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        def process_segment(asr_data_part):
            txt = asr_data_part.to_txt().replace("\n", "")
            sentences = split_by_llm(txt, use_cache=True)
            print(f"[+] 分段的句子提取完成，共 {len(sentences)} 句")
            return sentences
        all_sentences = list(executor.map(process_segment, asr_data_segments))
    all_sentences = [item for sublist in all_sentences for item in sublist]
    
    print(f"[+] 总共提取到 {len(all_sentences)} 句")

    # 基于LLM已经分段的句子，对ASR分段进行合并
    print("[+] 正在合并ASR分段基于句子列表...")
    merged_asr_data = merge_segments_based_on_sentences(asr_data, all_sentences)

    # 按开始时间排序合并后的分段(其实好像不需要)
    merged_asr_data.segments.sort(key=lambda seg: seg.start_time)
    final_asr_data = ASRData(merged_asr_data.segments)

    # 保存到SRT文件
    final_asr_data.to_srt(save_path=save_path)
    print(f"[+] 已保存合并后的SRT文件: {save_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="优化ASR分段处理脚本")
    parser.add_argument('--srt_path', type=str, required=True, help='输入的SRT文件路径')
    parser.add_argument('--save_path', type=str, required=True, help='输入的SRT文件路径')
    parser.add_argument('--num_threads', type=int, default=FIXED_NUM_THREADS, help='线程数量')
    args = parser.parse_args()

    # args.srt_path = "test_data/java.srt"
    # args.save_path = args.srt_path.replace(".srt", "_merged.srt")

    main(srt_path=args.srt_path, save_path=args.save_path, num_threads=args.num_threads)
