"""关于文本处理的一些工具脚本；"""
import os.path

import unicodedata
import jieba
from pypinyin import lazy_pinyin, Style, load_phrases_dict
from pypinyin.contrib.tone_convert import to_normal, to_tone, to_initials, to_finals


class TextFrontEnd:
    """处理文本的模块；"""
    def __init__(
            self,
            project_dir: str = "E:\\pytorchDeepLearning\\text_to_speech\\text_precess",
            phoneme_map_file: str = "",
    ):

        self.initial_jieba(project_dir)  # 加载jieba分词
        self.pinyin_map, self.initial_list, self.final_list = self.initial_pinyin(project_dir)  # 加载拼音、声母、韵母的映射
        self.phoneme_map = self.initial_phoneme_map(phoneme_map_file)  # 加载音素到ID的映射

        self.English_charactor_lower = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
                                        'h', 'i', 'j', 'k', 'l', 'm', 'n',
                                        'o', 'p', 'q', 'r', 's', 't',
                                        'u', 'v', 'w', 'x', 'y', 'z']

        self.English_charactor_upper = ['A', 'B', 'C', 'D', 'E', 'F', 'G',
                                        'H', 'I', 'J', 'K', 'L', 'M', 'N',
                                        'O', 'P', 'Q', 'R', 'S', 'T',
                                        'U', 'V', 'W', 'X', 'Y', 'Z']

        self.English_charactor_all = self.English_charactor_lower + self.English_charactor_upper

        # 分词级别的停顿：#1
        self.punctuation_pause1 = [
            "「", "」", "《", "》", "『", "』", "(", ")", "[", "]", "{", "}", "【", "】", "<", ">",  # 各种括号
            "—", "-", "~", "～",  # 各种横线
            "\"", "”", "“", "'", "‘", "’",  # 各种引号
            "·",  # 各种点
            "$", "♪", "|",  # 各种无意义符号
        ]

        # 逗号级别的停顿：#2
        self.punctuation_pause2 = [
            "，", ",",  # 中英文逗号
            "、",  # 中文顿号
            "；", ";",  # 中英文分号
        ]

        # 句号级别的停顿：#3
        self.punctuation_pause3 = [
            "。", ".",  # 中英文句号
            "…",  # 省略号
            "？", "?",  # 中英文问号
            "！", "!",  # 中英文叹号
            "：", ":",  # 中英文冒号
        ]

        self.punctuation_all = self.punctuation_pause1 + self.punctuation_pause2 + self.punctuation_pause3

        self.pause0 = "p0"  # 单个字之间的停顿
        self.pause1 = "p1"  # 词语之间的停顿
        self.pause2 = "p2"  # 逗号级别的停顿
        self.pause3 = "p3"  # 句号级别的停顿

        self.start_symbol = "sos"
        self.end_symbol = "eos"

        print(f"初始化完毕：文本前端模型")

    @staticmethod
    def combine_two_pause(pause_a, pause_b):
        """合并两个停顿标签，只保留最大的一个；例如，(#1, #2) -> #2；"""
        assert pause_a[-1] in ["0", "1", "2", "3"]
        assert pause_b[-1] in ["0", "1", "2", "3"]
        assert pause_a[:-1] == pause_b[:-1]
        return pause_a[:-1] + str(max(int(pause_a[-1]), int(pause_b[-1])))

    @staticmethod
    def initial_jieba(project_dir):
        """加载jieba词语；"""
        # 读取jieba分词词典
        jieba_add_word_list = []
        if os.path.exists(os.path.join(project_dir, "./data/jieba_dict")):
            jieba_dict_path = os.path.join(project_dir, "./data/jieba_dict")
        else:
            raise ValueError(f"找不到jieba分词词典")
        with open(jieba_dict_path, 'r', encoding='utf-8') as r1:
            for line in r1.readlines():
                line = str(line).strip()
                word, pinyin_list = line.split("\t", 1)
                jieba_add_word_list.append(word)
                load_phrases_dict({word: [[i] for i in pinyin_list.split(" ")]})

        # 加入到jieba热词中
        for word in jieba_add_word_list:
            jieba.add_word(word)

        # 删除这些词语
        jieba_del_word_list = [
            "用地", "和璃", "小说吧",
        ]
        for word in jieba_del_word_list:
            jieba.del_word(word)
        return

    @staticmethod
    def initial_pinyin(project_dir):
        """加载拼音列表"""
        # 加载：从拼音到声母&韵母的映射
        pinyin_map = {}
        initial_list = []
        final_list = []
        if os.path.exists(os.path.join(project_dir, "./data/pinyin_map")):
            pinyin_map_path = os.path.join(project_dir, "./data/pinyin_map")
        else:
            raise ValueError(f"找不到jieba分词词典")
        with open(pinyin_map_path, 'r', encoding='utf-8') as r1:
            for i, line in enumerate(r1.readlines()):
                line = str(line).strip()
                # 加载所有声母
                if i == 0:
                    for s in line.split("\t"):
                        if len(s) > 0:
                            initial_list.append(s)
                # 加载所有韵母&拼音
                else:
                    assert len(line.split("\t")) == len(initial_list)+1, f"第{i}行的拼音的数量，和声母的数量对不上；"
                    for j, s in enumerate(line.split("\t")):
                        s = str(s).replace("ü", "v")
                        # 加载所有韵母
                        if j == 0:
                            final_list.append(s)
                        # 加载所有拼音
                        elif len(s) > 0 and s != ["-"]:
                            for si in s.split("/"):
                                pinyin_map[si] = [initial_list[j-1], final_list[-1]]
        return pinyin_map, initial_list, final_list

    @staticmethod
    def initial_phoneme_map(phoneme_map_file, split_symbol=" "):
        """ 加载：音素到ID的映射 """
        if len(phoneme_map_file) < 1 or not os.path.exists(phoneme_map_file):
            return None
        else:
            phoneme_map = {}
            with open(phoneme_map_file, 'r', encoding='utf-8') as r1:
                for line in r1.readlines():
                    try:
                        phoneme, phoneme_id = line.strip().split(split_symbol)
                    except:
                        continue
                    phoneme_map[phoneme] = int(phoneme_id)
            return phoneme_map

    def is_alpha(self, char: str) -> bool:
        """判断一个字符是否是英文字母"""
        if char in self.English_charactor_all:
            return True
        else:
            return False

    @staticmethod
    def is_chinese(char: str) -> bool:
        """判断一个字符是否是中文"""
        if 'CJK' in unicodedata.name(char):
            return True
        else:
            return False

    def is_punctuation(self, char: str) -> bool:
        """判断一个字符是否是标点符号"""
        if char in self.punctuation_all or char in [self.start_symbol, self.end_symbol]:
            return True
        else:
            return False

    @staticmethod
    def jieba_cut(text_list) -> list:
        """
            jieba 分词，精确模式，用于 TTS 分词；
            text_list: list[ str or list[str] ]
            res: list[str]
        """
        if isinstance(text_list, str):
            text_list = [text_list]

        res = []
        for text in text_list:
            if isinstance(text, str):
                res += jieba.cut(text)
            elif isinstance(text, list):
                for t in text:
                    res += jieba.cut(t)

        return res

    def text2pinyin(self, text, add_pause0=False) -> list:
        """汉字转拼音，默认不加“#0”停顿；"""
        """ ref: https://github.com/mozillazg/python-pinyin """
        pinyin = lazy_pinyin(
            text,
            style=Style.TONE3,  # 拼音风格：数字在末尾：['yi1', 'shang5']
            neutral_tone_with_five=True,  # 使用 5 表示轻声
            tone_sandhi=True,  # 变调
        )
        if add_pause0 is False:
            return pinyin
        else:
            new_pinyin = []
            for i, p in enumerate(pinyin):
                if i != 0:
                    new_pinyin.append(self.pause0)  # 添加字级别的停顿
                new_pinyin += [p]
            return new_pinyin

    def text2phoneme(self, text, add_pause0=False) -> list:
        """汉字转音素：声母 + 韵母和声调；默认不加“#0”停顿；但不处理标点；"""
        pinyin = self.text2pinyin(text, add_pause0=add_pause0)

        phoneme = []
        for p in pinyin:
            if self.is_punctuation(p) or self.is_alpha(p) or p in [self.pause0, self.pause1, self.pause2, self.pause3]:
                phoneme.append(p)

            else:
                tone = p[-1]
                assert tone in ["1", "2", "3", "4", "5"], f'tone = {tone} not in ["1", "2", "3", "4", "5"]'

                initial, final = self.pinyin_map[p[:-1]]

                assert final[-1] not in ["1", "2", "3", "4", "5"], \
                    f'YunMu[-1] = {final[-1]} is already in ["1", "2", "3", "4", "5"]'

                final += tone

                phoneme.append(initial)
                phoneme.append(final)

        return phoneme

    def text2phoneme_ids(self, text, add_pause0=True) -> list[int]:
        """汉字转音素ID：默认添加“#0”停顿；"""
        assert isinstance(self.phoneme_map, dict), f"需要加载音素ID的映射文件"
        _, phoneme = self.text_processor(text)

        phoneme_ids = []
        for p in phoneme:
            assert p in self.phoneme_map, f"phoneme \"{p}\" not in phoneme_map"
            phoneme_ids.append(self.phoneme_map[p])

        return phoneme_ids

    def text_processor(self, input_text) -> tuple[list, list]:
        """文本前端模型：输入文本，输出音素；"""
        # 保证输入数据类型为 list；
        if isinstance(input_text, str):
            input_text = [input_text]

        # 汉字、标点 -> 分词
        word_list = []
        text_cache = ""
        for text in input_text:
            for char in text:
                if self.is_chinese(char):
                    text_cache += char
                else:
                    word_list += self.jieba_cut(text_cache)
                    text_cache = ""
                    word_list += [char]
            # 句末的分词
            if len(text_cache) > 0:
                word_list += self.jieba_cut(text_cache)

        # 分词 -> 音素 + 韵律
        new_word_list = []  # 为jieba分词添加#1、#2标签
        for word in word_list:
            if word in self.punctuation_pause1:
                new_word_list.append(self.pause1)
            elif word in self.punctuation_pause2:
                new_word_list.append(self.pause2)
            elif word in self.punctuation_pause3:
                new_word_list.append(self.pause3)
            else:
                new_word_list += [word, self.pause1]
        word_list = new_word_list

        # 检查韵律、合并连续的韵律标签
        new_word_list = []
        for word in word_list:
            if len(new_word_list) < 1:
                new_word_list += [word]
            elif word in [self.pause0, self.pause1, self.pause2, self.pause3] and new_word_list[-1] in [self.pause0, self.pause1, self.pause2, self.pause3]:
                max_phoneme = self.combine_two_pause(word, new_word_list[-1])
                new_word_list[-1] = max_phoneme
            else:
                new_word_list += [word]
        word_list = new_word_list

        # 保证句尾是 非韵律 + <eos>
        if word_list[-1] in [self.pause0, self.pause1, self.pause2, self.pause3]:
            word_list = word_list[:-1]
        word_list.append(self.end_symbol)

        # 保证句首是 <sos> + 非韵律
        if word_list[0] in [self.pause0, self.pause1, self.pause2, self.pause3]:
            word_list = word_list[1:]
        word_list = [self.start_symbol] + word_list

        # 中文 -> 拼音
        phoneme_list = []
        word_map = []
        for word in word_list:
            if word in [self.pause0, self.pause1, self.pause2, self.pause3, self.start_symbol, self.end_symbol]:
                phoneme_list.append(word)
                word_map.append([word, word])
            else:
                p = self.text2phoneme(word, add_pause0=True)
                phoneme_list += p
                word_map.append([word, " ".join(p)])

        # print(input_text)
        # print(word_list)
        # print(phoneme_list)
        # print()

        return word_map, phoneme_list

    def save_phoneme_map(self, save_path="./phoneme_map.txt"):
        """制作MFA所需的发音词典"""
        all_phoneme_list = [
            "pad",
            self.pause0, self.pause1, self.pause2, self.pause3,
            self.start_symbol, self.end_symbol
        ]  # 韵律标签

        all_phoneme_list += self.initial_list  # 声母

        for p in self.final_list:
            for i in range(5):
                all_phoneme_list.append(p + str(i+1))  # 韵母 + 声调

        with open(save_path, 'w', encoding='utf-8') as w1:
            for i, p in enumerate(all_phoneme_list):
                w1.write(" ".join([p, str(i)]) + "\n")

        return


if __name__ == "__main__":
    textFrontEnd = TextFrontEnd()

    s = "诺艾尔实在是太可爱了！人又温柔，还很有耐心。"

    word_list, phoneme_list = textFrontEnd.text_processor(s)
    print(word_list)
    print(phoneme_list)

    # 制作MFA所需的发音词典
    textFrontEnd.save_phoneme_map(save_path="./data/phoneme_map.txt")
