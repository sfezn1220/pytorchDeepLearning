""" 读取 label 文件；"""

import os
import json


def read_json_lists(
        input_file: str
):
    """ 读取 json 格式的 label 文件； """
    assert os.path.exists(input_file), f"label file not exists: {input_file}"

    lists = []
    with open(input_file, 'r', encoding='utf-8') as r1:

        for line in r1.readlines():
            line = line.strip()
            if len(line) < 1:
                continue

            json_line = json.loads(line)
            lists.append(json_line)

    return lists
