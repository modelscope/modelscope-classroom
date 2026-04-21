#!/usr/bin/env python3
"""Scan 大模型讲义 directory and generate content-index.json for the web viewer."""
import os
import json
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONTENT_DIR = os.path.join(SCRIPT_DIR, '..', '大模型讲义')
OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'content-index.json')


def parse_name(name):
    """Extract number and title from names like '01-有监督学习'."""
    m = re.match(r'(\d+)-(.*)', name)
    return (m.group(1), m.group(2)) if m else (None, name)


def count_files(children):
    """Count total leaf files in a tree."""
    total = 0
    for c in children:
        if c['type'] == 'file':
            total += 1
        elif 'children' in c:
            total += count_files(c['children'])
    return total


def scan_dir(path):
    """Recursively scan directory, return list of children nodes."""
    children = []
    try:
        entries = sorted(os.listdir(path))
    except OSError:
        return children

    files = [e for e in entries if e.endswith('.md') and os.path.isfile(os.path.join(path, e))]
    dirs = [e for e in entries if os.path.isdir(os.path.join(path, e)) and not e.startswith('.')]

    for fname in files:
        num, title = parse_name(fname[:-3])  # strip .md
        rel = os.path.relpath(os.path.join(path, fname), os.path.join(SCRIPT_DIR, '..'))
        children.append({
            'type': 'file',
            'id': num or fname,
            'title': title,
            'file': rel,
        })

    for dname in dirs:
        num, title = parse_name(dname)
        sub = scan_dir(os.path.join(path, dname))
        if sub:
            children.append({
                'type': 'group',
                'id': num or dname,
                'title': title,
                'children': sub,
            })

    return children


def main():
    chapters = []
    for entry in sorted(os.listdir(CONTENT_DIR)):
        full = os.path.join(CONTENT_DIR, entry)
        if not os.path.isdir(full) or entry.startswith('.'):
            continue
        num, title = parse_name(entry)
        if num is None:
            continue
        children = scan_dir(full)
        if children:
            chapters.append({
                'id': num,
                'title': title,
                'count': count_files(children),
                'children': children,
            })

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump({'chapters': chapters}, f, ensure_ascii=False, indent=2)

    print(f'Generated {len(chapters)} chapters -> {OUTPUT_FILE}')


if __name__ == '__main__':
    main()
