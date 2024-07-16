 # 从 爱奇艺 的开源数据集中，提取出所需数据
 # github: https://github.com/luxiangju-PersonAI/iCartoonFace?tab=readme-ov-file#Dataset
 
 
import os
import tqdm
import json
import random
import time
import pandas as pd
import requests
import datetime
from bs4 import BeautifulSoup


STOP_FLAG = "百度百科-验证"
STOP_TOKEN = "爬虫失败"


def write_to_excel(data, output_file: str, sheet_name: str = "Sheet1"):
    df = pd.DataFrame(data)
    df.to_excel(output_file, sheet_name=sheet_name)
    return


def from_html_get_class(url='https://baike.baidu.com/item/%E7%BB%87%E6%9C%AC%E6%B3%89/1590222?fromtitle=%E9%A3%8E%E7%A5%9E%E5%85%BD%E6%96%97%E5%A3%AB%E7%B2%BE%E7%A5%9E&fromid=7380995'):
    """ fork: https://zhuanlan.zhihu.com/p/136147927 """
    if url == "None":
        return ""

    header_list = [
        'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
        'Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; Acoo Browser 1.98.744; .NET CLR 3.5.30729)'
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; GTB5;"
    ]
    header = random.choice(header_list)

    headers = {'User-Agent': header}

    try:
        response = requests.get(url, headers=headers)
        #将一段文档传入BeautifulSoup的构造方法,就能得到一个文档的对象, 可以传入一段字符串
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()

        if STOP_FLAG in text:
            print(STOP_TOKEN)
            return STOP_TOKEN

        left_index, right_index = -1, -1
        left_find, right_find = False, False
        for i in range(len(text)):
            if text[i] == "《" and left_find is False:
                left_index = i
                left_find = True
            elif text[i] == "》" and right_find is False:
                right_index = i
                right_find = True
            else:
                pass

            if left_find is True and right_find is True:
                break

        if right_index > left_find:
            title = text[left_index:right_index].replace("《", "").replace("》", "")
            # print(f"find title {title}")
            return title
        else:
            return ""

    except Exception as e:
        print(f"Skip no title with error: {e}")
        return ""


class IqiyiDataset:
    def __init__(self):
        self.dir = "G:\\爱奇艺_动画截图"

        # stage 0: 读取所有类别
        self.input_lable_file = os.path.join(self.dir, "icartoonface_rectest_idInfo.txt")
        self.output_excel_file = os.path.join(self.dir, "0.所有类别的汇总.xlsx")

    def iqiyi_charactor_label_to_excel(self):
        """ 从 爱奇艺 开源数据的 类别label中，提取出 excel 文件； """

        charactor2id_list = []

        if os.path.exists(self.output_excel_file):
            print(f"load data from excel:")
            data = pd.ExcelFile(self.output_excel_file)
            df = data.parse("Sheet1")
            titles = df.loc[:, "title"]
            names = df.loc[:, "name"]
            ids = df.loc[:, "id"]
            urls = df.loc[:, "url"]
            for t, n, i, u in zip(titles, names, ids, urls):
                t = str(t) if str(t) != "nan" else ""
                item = {
                    "title": t,
                    "name": n,
                    "id": i,
                    "url": u,
                }
                charactor2id_list.append(item)
        else:
            print(f"load data from json file:")
            with open(self.input_lable_file, "r", encoding="utf-8") as r1:
                for line in r1.readlines():
                    line = line.strip()

                    try:
                        line = line.replace("\'", "\"").replace("None", "\"None\"")  # 适配 json.loads
                        line_json = json.loads(line)

                        name, id_i, url = line_json["name"], line_json["id"], line_json["url"]

                        if url == "None":
                            title = "空"
                        else:
                            title = ""
                        print(f"charactor: {name}, title: {title}")

                        item = {
                            "title": title,
                            "name": name,
                            "id": id_i,
                            "url": url,
                        }
                        charactor2id_list.append(item)

                    except:
                        print(f"load json ERROR: {line}")

        print(f"load data DONE!")
        return charactor2id_list

    def pachong(self, charactor2id_list):
        """ 爬取百度百科的数据； """
        new_charactor2id_list = []
        stop = False

        print(f"start pachong...")
        for item in tqdm.tqdm(charactor2id_list):
            if len(item["title"]) >= 1 or stop is True:
                pass
            else:
                title_may = from_html_get_class(item["url"])
                if title_may != STOP_TOKEN:
                    print(f"name = {item['name']}, title = {title_may}")
                    item["title"] = title_may if title_may != "" else "空"
                else:
                    stop = True
                    print(f"爬虫失败，暂停！")

                if stop is False:
                    t = random.choice(list(range(10, 30)))
                    time.sleep(t)

            new_charactor2id_list.append(item)

        return new_charactor2id_list

    def main(self):
        # 读取数据
        charactor2id_list = self.iqiyi_charactor_label_to_excel()

        # 爬虫
        new_charactor2id_list = self.pachong(charactor2id_list)

        # 保存
        write_to_excel(new_charactor2id_list, self.output_excel_file)
        return


if __name__ == "__main__":
    data = IqiyiDataset()
    data.main()
