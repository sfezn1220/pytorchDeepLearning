""" 读取 textgrid 文件；"""

import os
import textgrid


def read_one_textgrid(textgrid_file="", phoneme_list=[]) -> list[float]:
    """
    读取一条 textgrid 文件，并与音素序列进行对齐；
    :param textgrid_file: textgrid 文件的绝对路径；
    :param phoneme_list:  需要对齐的音素序列，例如：["n", "i2", "#0", "h", "ao3"]
    :return: list[float]: 每个音素的时长，秒；
    """

    # 初始化：每个音素的时长，秒：
    dur_list = [0.0 for _ in range(len(phoneme_list))]

    # 使用 textgrid 脚本读取对齐结果
    assert os.path.exists(textgrid_file) is True
    textGrid = textgrid.TextGrid()
    textGrid.read(textgrid_file)

    interval_phonemes = textGrid.tiers[1]

    for i, interval in enumerate(interval_phonemes):
        phoneme = interval.mark
        start_time = interval.minTime
        end_time = interval.maxTime
        print(f"{phoneme}, {end_time - start_time}")

    return dur_list


if __name__ == "__main__":
    # 测试单条textgrid文件的处理
    read_one_textgrid(
        textgrid_file="G:\Yuanshen\\3.loudnorm_16K_version-2.0-mfa_output\\Nuoaier\\Nuoaier-0007_1ia0c9sieyhmbfus8j390ibio3f7gyp_001725.TextGrid",
        phoneme_list="sos,t,ing1,p0,j,ian4,p1,l,e5,p1,m,a5,p3,sh,i4,p1,f,eng1,p0,sh,en2,p1,d,e5,p1,sh,eng1,p0,ii,in1,p3,m,ei3,p0,d,ao4,p1,uu,uo3,p1,x,in1,p0,f,an2,p0,ii,i4,p0,l,uan4,p1,d,e5,p1,sh,i2,p0,h,ou4,p2,t,ing1,p0,j,ian4,p1,f,eng1,p0,sh,eng1,p1,j,iu4,p1,n,eng2,p1,p,ing2,p0,j,ing4,p0,x,ia4,p0,l,ai2,eos".split(","),
    )
