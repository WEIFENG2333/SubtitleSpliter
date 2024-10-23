import re
import openai
import os
import json
import hashlib

os.environ['OPENAI_BASE_URL'] = "https://api.ephone.chat/v1"
os.environ["OPENAI_API_KEY"] = "msuqsbr4Ano2cV1GDk0AIaUS1MXS3vukQF0eI7Cble8CgF7Q"
MODEL= "gpt-4o-mini"

client = openai.OpenAI()

system_prompt = """
你是一名字幕断句修复专家，擅长将没有断句的文本，进行断句成一句句文本，断句文本之间用<br>隔开。

要求：
1. 每个断句文本字数（单词数）不超过12。。
2. 不按照完整的句子断句，只需按照语义进行分割，例如在"而"、"的"、"在"、"和"、"so"、"but"等词或者语气词后进行断句。
3. 不要修改原句的任何内容，也不要添加任何内容，你只需要每个断句文本之间添加<br>隔开。
4. 直接返回断句后的文本，不要返回任何其他说明内容。

输入：
大家好今天我们带来的3d创意设计作品是禁制演示器我是来自中山大学附属中学的方若涵我是陈欣然我们这一次作品介绍分为三个部分第一个部分提出问题第二个部分解决方案第三个部分作品介绍当我们学习进制的时候难以掌握老师教学 也比较抽象那有没有一种教具或演示器可以将进制的原理形象生动地展现出来


输出：
大家好<br>今天我们带来的<br>3d创意设计作品是禁制演示器<br>我是来自中山大学附属中学的方若涵<br>我是陈欣然<br>我们这一次作品介绍分为三个部分<br>第一个部分提出问题<br>第二个部分解决方案<br>第三个部分作品介绍<br>当我们学习进制的时候难以掌握<br>老师教学也比较抽象<br>那有没有一种教具或演示器<br>可以将进制的原理形象生动地展现出来
"""

def get_cache_key(text, model):
    return hashlib.md5(f"{text}_{model}".encode()).hexdigest()

def get_cache(text, model):
    cache_key = get_cache_key(text, model)
    cache_file = f"cache/{cache_key}.json"
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def set_cache(text, model, result):
    cache_key = get_cache_key(text, model)
    cache_file = f"cache/{cache_key}.json"
    os.makedirs("cache", exist_ok=True)
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)

def split_by_llm(text):
    cached_result = get_cache(text, MODEL)
    if cached_result:
        return cached_result

    prompt = f"请你对下面句子使用<br>进行分割：\n{text}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}],
        temperature=0.1
    )
    print(response)
    result = response.choices[0].message.content
    # 去除多个换行符
    result = re.sub(r'\n+', '', result)
    split_result = result.split("<br>")
    
    set_cache(text, MODEL, split_result)
    return split_result


if __name__ == "__main__":
    text = "大家好我叫杨玉溪来自有着良好音乐氛围的福建厦门自记事起我眼中的世界就是朦胧的童话书是各色杂乱的线条电视机是颜色各异的雪花小伙伴是只听其声不便骑行的马赛克后来我才知道这是一种眼底黄斑疾病虽不至于失明但终身无法治愈"
    print(split_by_llm(text))


