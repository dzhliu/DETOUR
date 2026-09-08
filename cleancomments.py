#!/usr/bin/env python3

import os
import re
import shutil
from pathlib import Path

def remove_comments_from_py_file(file_path, backup=True):
    """
    try:

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if not content.strip():
            return True

        cleaned_content = remove_comments(content)

        if cleaned_content == content:
            return True

        if backup:
            backup_path = str(file_path) + '.bak'
            shutil.copy2(file_path, backup_path)
            print(f"✓ 已创建备份: {backup_path}")

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)

        print(f"✓ 已处理: {file_path}")
        return True

    except Exception as e:
        print(f"✗ 处理失败 {file_path}: {str(e)}")
        return False

def remove_comments(code):
    """
    lines = code.split('\n')
    result = []
    in_multiline_comment = False
    in_string = False
    string_char = None

    for line in lines:

        if in_multiline_comment:

            end_pos = line.find('"""')
            if end_pos == -1:
                end_pos = line.find("'''")

            if end_pos != -1:

                in_multiline_comment = False

                remaining = line[end_pos + 3:]
                if remaining.strip():

                    remaining_cleaned = remove_comments(remaining)
                    if remaining_cleaned.strip():
                        result.append(remaining_cleaned)
            continue

        new_line = ''
        i = 0
        while i < len(line):
            char = line[i]

            if char in ('"', "'") and not in_multiline_comment:

                if i + 2 < len(line) and line[i:i+3] in ('"""', "'''"):

                    if not in_string:
                        in_multiline_comment = True
                        new_line += line[i:i+3]
                        i += 3
                        continue
                else:

                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:

                        if i > 0 and line[i-1] != '\\':
                            in_string = False
                            string_char = None
                    new_line += char
                    i += 1
                    continue

            # 在字符串内
            if in_string or in_multiline_comment:
                new_line += char
                i += 1
                continue

            # 处理单行注释
            if char == '

                if i == 0 and line.startswith('#!'):
                    new_line += line[i:]
                    break

                break

            new_line += char
            i += 1

        if in_multiline_comment and new_line.strip():
            result.append(new_line)
        elif not in_multiline_comment:

            if new_line.strip() or (result and result[-1].strip()):
                result.append(new_line.rstrip())

    cleaned_lines = []
    for line in result:
        if line.strip() or (cleaned_lines and cleaned_lines[-1].strip()):
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)

def process_directory(directory='.', backup=True, dry_run=False):
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"错误: 目录不存在 {directory}")
        return

    print(f"开始处理目录: {directory.absolute()}")
    print("=" * 60)

    py_files = []
    for root, dirs, files in os.walk(directory):

        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['venv', 'env', '__pycache__']]

        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                py_files.append(file_path)

    if not py_files:
        print("未找到Python文件")
        return

    print(f"找到 {len(py_files)} 个Python文件")

    if dry_run:
        print("\n[试运行模式] 将处理以下文件:")
        for file_path in py_files:
            print(f"  - {file_path}")
        print(f"\n共 {len(py_files)} 个文件")
        return

    success_count = 0
    for file_path in py_files:
        if remove_comments_from_py_file(file_path, backup):
            success_count += 1

    print("=" * 60)
    print(f"处理完成！成功处理 {success_count}/{len(py_files)} 个文件")

    if backup:
        print("提示: 备份文件以 .bak 结尾，确认无误后可手动删除")

def main():
    """
    import argparse

    parser = argparse.ArgumentParser(description='去除Python文件中的所有注释')
    parser.add_argument('-d', '--directory', default='.',
                       help='要处理的目录路径 (默认: 当前目录)')
    parser.add_argument('--no-backup', action='store_true',
                       help='不创建备份文件')
    parser.add_argument('--dry-run', action='store_true',
                       help='试运行模式，只显示将要处理的文件')

    args = parser.parse_args()

    process_directory(
        directory=args.directory,
        backup=not args.no_backup,
        dry_run=args.dry_run
    )

if __name__ == '__main__':
    main()
