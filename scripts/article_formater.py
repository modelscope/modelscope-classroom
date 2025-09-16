import hashlib
import re
import shutil
from concurrent.futures.thread import ThreadPoolExecutor
from urllib.parse import urlparse

import requests
import os
from pathlib import Path

from openai import OpenAI


def download_media(url, save_dir, filename):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    file_path = os.path.join(save_dir, filename)

    parsed_url = urlparse(url)
    media_name = os.path.basename(parsed_url.path)

    if not media_name or '.' not in media_name:
        content_type = response.headers.get('content-type', '')
        if 'image' in content_type or 'video' in content_type or 'audio' in content_type:
            ext = content_type.split('/')[-1]
        else:
            raise
    else:
        ext = media_name.split('.')[-1]

    file_path = file_path + '.' + ext
    with open(file_path + '.' + ext, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    return file_path


system = """You are a professional document translator who can translate from source language to target language. Your source language is 中文, and your target language is English.

What follows is a portion of an academic article that you need to translate according to academic context. You need to pay attention to:

1. Ensure translation accuracy and do not omit any sentences or paragraphs
2. Lists and tables also need to be translated, maintaining their usability after translation
3. Ensure formulas, image links, etc. are output as-is
4. If a proper noun cannot be translated, or loss its meaning after translation, keep it unchanged
5. Your output should not contain ```, nor should it include any summary of the input paragraphs. **Your responsibility is only translation** - do not output extraneous symbols or statements

Now begin:
"""


def contains_source_lang(content):
    return bool(re.search(r'[\u4e00-\u9fff]', content))


def process_markdown_media(
        content,
        save_dir,
        article_name,
) -> str:
    media_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'

    url_signatures = os.environ['URL_SIGNATURE'].split(',')
    url_signatures = [sig.strip() for sig in url_signatures if sig.strip()]

    def replace_media(match):
        alt_text = match.group(1)
        url = match.group(2)
        if not any([sig in url for sig in url_signatures]):
            return f'![{alt_text}]({url})'

        local_media = download_media(url, save_dir, article_name + hashlib.md5(url.encode("utf-8")).hexdigest()[:8])
        local_media = local_media.replace(os.path.sep, '/')
        github_media = f'https://github.com/modelscope/modelscope-classroom/blob/main/{local_media}'
        return f'![{alt_text}]({github_media})'
    return re.sub(media_pattern, replace_media, content)


def do_translate(content):
    if not contains_source_lang(content):
        return content
    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': content},
    ]
    client = OpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        base_url=f'https://dashscope.aliyuncs.com/compatible-mode/v1',
    )
    resp = client.chat.completions.create(model='claude-sonnet-4-20250514', messages=messages, max_tokens=65536, temperature=0.3)
    return resp.choices[0].message.content


def merge_blocks(blocks):
    if not blocks:
        return []

    merged = []
    current_block = ""

    for block in blocks:
        if len(current_block) + len(block) <= 2048:
            current_block += (block + '\n')
        else:
            if current_block:
                merged.append(current_block)
            current_block = block

    if current_block:
        merged.append(current_block)

    return merged


def translate(contents):
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(do_translate, content) for content in contents]
        return [future.result() for future in futures]


def translate_md(source_file, article_type='Blogs', article_group='Articles', en_title=None):
    if not source_file.endswith('.md'):
        return

    os.makedirs('_temp', exist_ok=True)
    with open(source_file, 'r') as fr:
        content = fr.read()
        blocks = content.split('\n')
        _final_blocks = []
        title = None
        for block in blocks[:3]:
            if block.startswith('# '):
                title = block.split('# ')[1].strip()
                break
        assert title is not None
        if en_title is None:
            en_title = do_translate(title).replace(' ', '-')
        article_path = os.path.join(article_type, article_group, en_title)
        os.makedirs(article_path, exist_ok=True)
        zh_target_file = os.path.join(article_path, 'report.md')
        en_target_file = os.path.join(article_path, 'report_en.md')
        assets_path = os.path.join(article_path, 'resources')

        if not os.path.exists(en_target_file) or not os.path.exists(zh_target_file):
            for block in blocks:
                block = block.replace(' ', ' ')
                block = process_markdown_media(block, assets_path, '')
                if '|' in block and _final_blocks and '|' in _final_blocks[-1]:
                    _final_blocks[-1] = _final_blocks[-1] + '\n' + block
                else:
                    _final_blocks.append(block)
            blocks = _final_blocks

        blocks = merge_blocks(blocks)

        if not os.path.exists(en_target_file):
            en_blocks = translate(blocks)
        else:
            en_blocks = blocks

    en_temp_file = os.path.join('_temp', 'report_en.md')
    if not os.path.exists(en_target_file):
        with open(en_temp_file, 'w') as fw:
            final_content = '\n'.join(en_blocks)
            fw.write(final_content)
        shutil.move(en_temp_file, en_target_file)

    zh_temp_file = os.path.join('_temp', 'report.md')
    if not os.path.exists(zh_target_file):
        with open(zh_temp_file, 'w') as fw:
            final_content = '\n'.join(blocks)
            fw.write(final_content)
        shutil.move(zh_temp_file, zh_target_file)


def run_md():
    shutil.rmtree('_temp', ignore_errors=True)
    md_file = '/Users/tastelikefeet/Downloads/万字长文深度解析最新Deep Research技术：前沿架构、核心技术与未来展望.md'
    translate_md(md_file, en_title='Deep-Research-Survey')


if __name__ == '__main__':
    run_md()