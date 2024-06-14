import os


for file in ['B.魔搭社区和LLM大模型基础知识.md',
             'C.提示词工程-prompt engineering.md',
             'I.LLM和多模态模型高效推理实践.md',
             'K.大模型自动评估理论和实战--LLM Automatic Evaluation.md']:
    with open(file, 'r') as f:
        with open(file + '.new', 'w') as f2:
            for line in f.readlines():
                if '![image](' in line and 'alidocs.oss-cn-zhangjiakou.aliyuncs.com' in line:
                    idx = line.find('![image](')
                    before = line[:idx+len('![image](')]
                    part = line[idx+len('![image]('):]
                    idx = part.find(')')
                    link = part[:idx]
                    after = part[idx:]
                    idx = link.rfind('/')
                    filename = link[idx+1:]
                    os.system(f'curl {link} > resources/{filename}')
                    line = before + f'resources/{filename}' + after
                f2.write(line)

