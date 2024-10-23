import hashlib
import json
import os
import re
from typing import List, Optional
import openai
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# 使用环境变量
os.environ['OPENAI_BASE_URL'] = os.getenv('OPENAI_BASE_URL')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# ... 其余代码保持不变 ...
# 常量定义
MODEL = "gpt-4o-mini"
CACHE_DIR = "cache"

# 初始化OpenAI客户端
client = openai.OpenAI()

# 系统提示信息
SYSTEM_PROMPT = """
你是一名字幕断句修复专家，擅长将没有断句的文本，进行断句成一句句文本，断句文本之间用<br>隔开。

要求：
1. 对于中文每个断句文本总字数不超过12；对于英文每个断句单词(word)总数目不超过12。
2. 不按照完整的句子断句，只需按照语义进行分割，例如在"而"、"的"、"在"、"和"、"so"、"but"等词或者语气词后进行断句。
3. 不要修改原句的任何内容，也不要添加任何内容，你只需要每个断句文本之间添加<br>隔开。
4. 直接返回断句后的文本，不要返回任何其他说明内容。

输入：
大家好今天我们带来的3d创意设计作品是禁制演示器我是来自中山大学附属中学的方若涵我是陈欣然我们这一次作品介绍分为三个部分第一个部分提出问题第二个部分解决方案第三个部分作品介绍当我们学习进制的时候难以掌握老师教学 也比较抽象那有没有一种教具或演示器可以将进制的原理形象生动地展现出来
输出：
大家好<br>今天我们带来的<br>3d创意设计作品是禁制演示器<br>我是来自中山大学附属中学的方若涵<br>我是陈欣然<br>我们这一次作品介绍分为三个部分<br>第一个部分提出问题<br>第二个部分解决方案<br>第三个部分作品介绍<br>当我们学习进制的时候难以掌握<br>老师教学也比较抽象<br>那有没有一种教具或演示器<br>可以将进制的原理形象生动地展现出来


输入：
the upgraded claude sonnet is now available for all users developers can build with the computer use beta on the anthropic api amazon bedrock and google cloud’s vertex ai the new claude haiku will be released later this month
输出：
the upgraded claude sonnet is now available for all users<br>developers can build with the computer use beta<br>on the anthropic api amazon bedrock and google cloud’s vertex ai<br>the new claude haiku will be released later this month
"""

def get_cache_key(text: str, model: str) -> str:
    """
    生成缓存键值
    """
    return hashlib.md5(f"{text}_{model}".encode()).hexdigest()

def get_cache(text: str, model: str) -> Optional[List[str]]:
    """
    从缓存中获取断句结果
    """
    cache_key = get_cache_key(text, model)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError):
            return None
    return None

def set_cache(text: str, model: str, result: List[str]) -> None:
    """
    将断句结果设置到缓存中
    """
    cache_key = get_cache_key(text, model)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)
    except IOError:
        pass

def split_by_llm(text: str, use_cache: bool = False) -> List[str]:
    """
    使用LLM进行文本断句
    """
    if use_cache:
        cached_result = get_cache(text, MODEL)
        if cached_result:
            print(f"[+] 从缓存中获取结果: {cached_result}")
            return cached_result

    prompt = f"请你对下面句子使用<br>进行分割：\n{text}"

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        result = response.choices[0].message.content
        # 清理结果中的多余换行符
        result = re.sub(r'\n+', '', result)
        split_result = [segment.strip() for segment in result.split("<br>") if segment.strip()]
        
        set_cache(text, MODEL, split_result)
        return split_result
    except Exception as e:
        print(f"[!] 请求LLM失败: {e}")
        return []

if __name__ == "__main__":
    sample_text = (
        "大家好我叫杨玉溪来自有着良好音乐氛围的福建厦门自记事起我眼中的世界就是朦胧的童话书是各色杂乱的线条电视机是颜色各异的雪花小伙伴是只听其声不便骑行的马赛克后来我才知道这是一种眼底黄斑疾病虽不至于失明但终身无法治愈"
    )
    sentences = split_by_llm(sample_text, use_cache=True)
    print(sentences)
