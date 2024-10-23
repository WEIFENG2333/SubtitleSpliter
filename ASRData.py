import re
from typing import List


class ASRDataSeg:
    def __init__(self, text, start_time, end_time):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time

    def to_srt_ts(self) -> str:
        """Convert to SRT timestamp format"""
        return f"{self._ms_to_srt_time(self.start_time)} --> {self._ms_to_srt_time(self.end_time)}"

    @staticmethod
    def _ms_to_srt_time(ms) -> str:
        """Convert milliseconds to SRT time format (HH:MM:SS,mmm)"""
        total_seconds, milliseconds = divmod(ms, 1000)
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{int(milliseconds):03}"

    def to_lrc_ts(self) -> str:
        """Convert to LRC timestamp format"""
        return f"[{self._ms_to_lrc_time(self.start_time)}]"

    def _ms_to_lrc_time(self, ms) -> str:
        seconds = ms / 1000
        minutes, seconds = divmod(seconds, 60)
        return f"{int(minutes):02}:{seconds:.2f}"

    @property
    def transcript(self) -> str:
        """Return segment text"""
        return self.text

    def __str__(self) -> str:
        return f"ASRDataSeg({self.text}, {self.start_time}, {self.end_time})"


class ASRData:
    def __init__(self, segments: List[ASRDataSeg]):
        self.segments = segments

    def __iter__(self):
        return iter(self.segments)

    def has_data(self) -> bool:
        """Check if there are any utterances"""
        return len(self.segments) > 0

    def to_txt(self) -> str:
        """Convert to plain text subtitle format (without timestamps)"""
        return "\n".join(seg.transcript for seg in self.segments)

    def to_srt(self, save_path=None) -> str:
        """Convert to SRT subtitle format"""
        srt_text = "\n".join(
            f"{n}\n{seg.to_srt_ts()}\n{seg.transcript}\n"
            for n, seg in enumerate(self.segments, 1))
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(srt_text)
        return srt_text

    def to_lrc(self) -> str:
        """Convert to LRC subtitle format"""
        return "\n".join(
            f"{seg.to_lrc_ts()}{seg.transcript}" for seg in self.segments
        )

    def to_ass(self) -> str:
        """Convert to ASS subtitle format"""
        raise NotImplementedError("ASS format conversion not implemented yet")

    def to_json(self) -> dict:
        result_json = {}
        for i, segment in enumerate(self.segments, 1):
            # 检查是否有换行符
            if "\n" in segment.text:
                original_subtitle, translated_subtitle = segment.text.split("\n")
            else:
                original_subtitle, translated_subtitle = segment.text, ""

            result_json[str(i)] = {
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "original_subtitle": original_subtitle,
                "translated_subtitle": translated_subtitle
            }
        return result_json

    def merge_segments(self, start_index: int, end_index: int, merged_text: str = None):
            """合并从 start_index 到 end_index 的段（包含）。"""
            if start_index < 0 or end_index >= len(self.segments) or start_index > end_index:
                raise IndexError("无效的段索引。")
            merged_start_time = self.segments[start_index].start_time
            merged_end_time = self.segments[end_index].end_time
            if merged_text is None:
                merged_text = ''.join(seg.text for seg in self.segments[start_index:end_index+1])
            merged_seg = ASRDataSeg(merged_text, merged_start_time, merged_end_time)
            # 替换 segments[start_index:end_index+1] 为 merged_seg
            # self.segments[start_index:end_index+1] = [merged_seg]
            return merged_seg

    def merge_with_next_segment(self, index: int) -> None:
        """合并指定索引的段与下一个段。"""
        if index < 0 or index >= len(self.segments) - 1:
            raise IndexError("索引超出范围或没有下一个段可合并。")
        current_seg = self.segments[index]
        next_seg = self.segments[index + 1]

        # 合并文本
        merged_text = f"{current_seg.text} {next_seg.text}"
        merged_start_time = current_seg.start_time
        merged_end_time = next_seg.end_time
        merged_seg = ASRDataSeg(merged_text, merged_start_time, merged_end_time)

        # 替换当前段为合并后的段
        self.segments[index] = merged_seg
        # 删除下一个段
        del self.segments[index + 1]

    def __str__(self):
        return self.to_txt()


def from_srt(srt_str: str) -> 'ASRData':
    """
    从SRT格式的字符串创建ASRData实例。

    :param srt_str: 包含SRT格式字幕的字符串。
    :return: 解析后的ASRData实例。
    """
    segments = []
    srt_time_pattern = re.compile(
        r'(\d{2}):(\d{2}):(\d{1,2})[.,](\d{3})\s-->\s(\d{2}):(\d{2}):(\d{1,2})[.,](\d{3})'
    )

    for block in re.split(r'\n\s*\n', srt_str.strip()):
        lines = block.splitlines()
        if len(lines) < 3:
            raise ValueError(f"无效的SRT块格式: {block}")

        match = srt_time_pattern.match(lines[1])
        if not match:
            raise ValueError(f"无效的时间戳格式: {lines[1]}")

        time_parts = list(map(int, match.groups()))
        start_time = sum([
            time_parts[0] * 3600000,
            time_parts[1] * 60000,
            time_parts[2] * 1000,
            time_parts[3]
        ])
        end_time = sum([
            time_parts[4] * 3600000,
            time_parts[5] * 60000,
            time_parts[6] * 1000,
            time_parts[7]
        ])

        text = '\n'.join(lines[2:]).strip()
        segments.append(ASRDataSeg(text, start_time, end_time))

    return ASRData(segments)

def from_vtt(vtt_str: str) -> 'ASRData':
    """
    从VTT格式的字符串创建ASRData实例, 去除不必要的样式和HTML信息。

    :param vtt_str: 包含WebVTT格式字幕的字符串。
    :return: 解析后的ASRData实例。
    """
    segments = []
    # 正则表达式匹配时间码行
    vtt_time_pattern = re.compile(
        r'(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s-->\s(\d{2}):(\d{2}):(\d{2})\.(\d{3})'
    )
    
    # 分割字幕块，VTT 使用两个换行符分隔块
    blocks = re.split(r'\n\s*\n', vtt_str.strip())
    # for i in range(len(blocks)):
    #     print(f"block {i}: {repr(blocks[i])}")
    
    for block in blocks:
        lines = block.splitlines()
        if not lines:
            continue
        # 跳过文件头和注释块
        if lines[0].startswith('WEBVTT') or lines[0].startswith('NOTE'):
            continue
        # 如果块以数字开头（可选标识符），则跳过第一行
        if re.match(r'^\d+$', lines[0]):
            lines = lines[1:]
        if len(lines) < 2:
            continue  # 无效的块
        
        # 匹配时间码
        match = vtt_time_pattern.match(lines[0])
        if not match:
            continue  # 无效的时间码格式
        
        time_parts = list(map(int, match.groups()))
        start_time = (
            time_parts[0] * 3600000 +
            time_parts[1] * 60000 +
            time_parts[2] * 1000 +
            time_parts[3]
        )
        end_time = (
            time_parts[4] * 3600000 +
            time_parts[5] * 60000 +
            time_parts[6] * 1000 +
            time_parts[7]
        )
        
        # 合并文本行并去除样式和HTML标签
        raw_text = ' '.join(lines[1:]).strip()
        # 去除尖括号内的内容（如样式标签和时间戳标签）
        clean_text = re.sub(r'<[^>]+>', '', raw_text)
        # 去除多余的空格
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        segments.append(ASRDataSeg(clean_text, start_time, end_time))
    
    return ASRData(segments)


if __name__ == '__main__':
    # 测试
    from pathlib import Path
    # vtt_file_path = r"E:\GithubProject\VideoCaptioner\app\work_dir\Setting the record straight\subtitle\original_subtitle.en.vtt"
    # asr_data = from_vtt(Path(vtt_file_path).read_text(encoding="utf-8"))
    srt_file_path = r"E:\GithubProject\VideoCaptioner\app\core\output\subtitles_fix.srt"
    asr_data = from_srt(Path(srt_file_path).read_text(encoding="utf-8"))

    print(asr_data.to_txt())
    # pass
    # asr_data = ASRData(seg)
    # Uncomment to test different formats:
    # print(asr_data.to_srt())
    # print(asr_data.to_lrc())
    # print(asr_data.to_txt())
    # print(asr_data.to_json())
    # print(asr_data.to_json())




