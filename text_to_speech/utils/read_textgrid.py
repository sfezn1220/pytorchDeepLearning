""" 读取 textgrid 文件；"""

import os
import tqdm
import textgrid


# 这些是韵律标签
punctuation_token_list = [
    "#0", "#1", "#2", "#3", "<sos>", "<eos>",
    "p0", "p1", "p2", "p3", "sos", "eos",
]


def read_one_textgrid(textgrid_file: str = "", phoneme_list: list = []) -> list[list[str, float]]:
    """
    读取一条 textgrid 文件，并与音素序列进行对齐；
    :param textgrid_file: textgrid 文件的绝对路径；
    :param phoneme_list:  需要对齐的音素序列，例如：["n", "i2", "#0", "h", "ao3"]
    :return: list[float]: 每个音素的时长，秒；
    """

    # 初始化：每个音素的时长，秒：
    dur_list = []

    # 使用 textgrid 工具，读取对齐结果
    assert os.path.exists(textgrid_file) is True
    textGrid = textgrid.TextGrid()
    textGrid.read(textgrid_file)

    interval_phonemes = textGrid.tiers[1]

    # 提取出音素，及对应的时长，秒
    blank_phoneme_cnt = 0
    for i, interval in enumerate(interval_phonemes):
        phoneme = interval.mark
        start_time = float(interval.minTime)
        end_time = float(interval.maxTime)
        dur_list.append([phoneme, abs(start_time - end_time)])

        if len(phoneme) < 1:
            blank_phoneme_cnt += 1  # 统计：有多少个MFA对齐出的空字符

    # 删除掉所有的MFA对齐的空字符
    for _ in range(blank_phoneme_cnt):
        for i, [p, dur] in enumerate(dur_list):
            if len(p) < 1:  # 如果是空字符
                if i-1 >= 0 and dur_list[i-1][0] in punctuation_token_list:
                    dur_list[i-1][-1] += dur_list[i][-1]
                    dur_list.pop(i)
                    break
                elif i+1 <= len(dur_list)-1 and dur_list[i+1][0] in punctuation_token_list:
                    dur_list[i+1][-1] += dur_list[i][-1]
                    dur_list.pop(i)
                    break

    # 验证：音素长度是否一致，如果给定音素的话：
    if len(phoneme_list) > 1:
        assert len(dur_list) == len(phoneme_list)

        p_mfa = [p for [p, dur] in dur_list]
        assert p_mfa == phoneme_list

    return dur_list


def read_all_textgrid(textgred_dir: str = "", uttid_useful_list: list = []) -> dict:
    """读取这个文件夹内的所有textgrid文件；返回字典：key=uttid, value=[[phoneme1, phoneme2, ...], [dur1, dur2, ...]]；"""

    uttid2textgrid = {}
    print(f"loading all textgrid files...")

    for spk in tqdm.tqdm(os.listdir(textgred_dir)):
        if not os.path.isdir(os.path.join(textgred_dir, spk)):
            continue
        for file in os.listdir(os.path.join(textgred_dir, spk)):
            if not file.endswith(".TextGrid"):  # 只读取textgrid文件
                continue
            uttid = file.replace(".TextGrid", "")
            full_path = os.path.join(textgred_dir, spk, file)

            # 检查是不是所需要的 uttid
            if len(uttid_useful_list) > 0 and uttid not in uttid_useful_list:
                continue

            phoneme_dur_list = read_one_textgrid(full_path)  # 读取一条textgrid文件，删除空字符

            phoneme_list = [p for p, d in phoneme_dur_list]
            dur_list = [d for p, d in phoneme_dur_list]

            if uttid not in uttid2textgrid:
                uttid2textgrid[uttid] = [phoneme_list, dur_list]
            else:
                print(f"重复的 uttid: {uttid}")

    return uttid2textgrid


def read_spk_uttid_textgrid(textgred_dir: str = "", spk_uttid_list: list = []) -> dict:
    """ 根据指定的 spk 和 uttid，找出对应的 textgrid 文件； """

    uttid2textgrid = {}
    print(f"loading all textgrid files...")

    for spk, uttid in tqdm.tqdm(spk_uttid_list):

        full_path = os.path.join(textgred_dir, spk, uttid + ".TextGrid")
        if not os.path.exists(full_path):
            continue

        phoneme_dur_list = read_one_textgrid(full_path)  # 读取一条textgrid文件，删除空字符

        phoneme_list = [p for p, d in phoneme_dur_list]
        dur_list = [d for p, d in phoneme_dur_list]

        if uttid not in uttid2textgrid:
            uttid2textgrid[uttid] = [phoneme_list, dur_list]
        else:
            print(f"重复的 uttid: {uttid}")

    return uttid2textgrid


if __name__ == "__main__":
    # 测试单条textgrid文件的处理
    read_one_textgrid(
        textgrid_file="G:\Yuanshen\\3.loudnorm_16K_version-2.0-mfa_output\\Nuoaier\\Nuoaier-0007_1ia0c9sieyhmbfus8j390ibio3f7gyp_001725.TextGrid",
        phoneme_list="sos,t,ing1,p0,j,ian4,p1,l,e5,p1,m,a5,p3,sh,i4,p1,f,eng1,p0,sh,en2,p1,d,e5,p1,sh,eng1,p0,ii,in1,p3,m,ei3,p0,d,ao4,p1,uu,uo3,p1,x,in1,p0,f,an2,p0,ii,i4,p0,l,uan4,p1,d,e5,p1,sh,i2,p0,h,ou4,p2,t,ing1,p0,j,ian4,p1,f,eng1,p0,sh,eng1,p1,j,iu4,p1,n,eng2,p1,p,ing2,p0,j,ing4,p0,x,ia4,p0,l,ai2,eos".split(","),
    )
