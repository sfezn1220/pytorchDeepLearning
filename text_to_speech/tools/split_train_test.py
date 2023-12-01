""" 输入 label文件、划分训练集、测试集； """

import os
import tqdm


test_basename_sets = [
    "Abeiduo-0000_kw7rtfpo4dmxbdmgx4ryfezznc6oo4i",
    "Babala-0000_hxj4a3e2aw2oundfm7ddctfs8nqzake",
    "Bachongshenzi-0000_j8bsdnkvi7nnv5fygudeymouxoir04p",
    "Bannite-0000_jna16rusizvz71pjygrsm2ihhzxmy7o",
    "Chongyun-0000_sxtn0zmwf70u62sdkljoz4xovcrrzks9",
    "Dadalian-0000_fmrlyvhwpxu082czmruxa4kgdoew7c5",
    "Diaona-0000_f032apxfqa231wdgb2ugupgntreo8zr",
    "Diluke-0000_k33bresfktz04phbhtkallqz234oea3",
    "Feixieer-0000_nfturpxw10en3nb99amvjt2r3468qdg",
    "Fengyuanwanye-0000_pfu7mnyysrat310mhrm6az5m6e9ymw4",
    "Ganyu-0000_8afe4wh6lkcmpoylg9f4njk1c2jt46t",
    "Huanglongyidou-0000_7jr19i05efuan5wy4p9sv9r3syr3gqh",
    "Hutao-0000_73ne3fbhw4emno2z3ss6wf7zjdmpph9",
    "Jiutiaoshaluo-0001_fy31uukykatzjzpugc207oqsk0mp63r",
    "Keli-0000_icnus2c6r8piuvz8k9uz2i5tt0w9ydb",
    "Keqing-0000_pwfdclndgm1ko3aldw7exakv3m1okz0",
    "Leidianjiangjun-0005_gbk463w5dkebim9fe1a1szqrgryl93r",
    "Lisha-0002_s9z1c15d09jktqmzx68sg2gki33bi20",
    "Luoshaliya-0011_4c72ikrmyigd5t7uexxrd4fs236k7pl",
    "Mona-0000_oigsdo9qjdx0y69kibx756x3z6fh106",
    "Ningguang-0000_aop2mtdmoe6nv80quopzx6upqn192fl",
    "Nuoaier-0000_h4idjj9ayd0qo12u6rghfjm7a75n71l",
    "Qin-0000_2fb1097nmz3pylf9rni0fs6l6g92xnm",
    "Qiqi-0000_o90h18wb0xuivuw6cw7wtmp8wotjzau",
    "Shanhugongxinhai-0000_0k65v3e7bjw1ka7oqzhthk4lg3t6spn",
    "Shatang-0041_5frhvfx5egkw3icidyomhhsprkhyn0q",
    "Shenhe-0000_3d4fhyppxc49nztnowjmtjubvs58amm",
    "Shenlilinghua-0000_hx23f7cf96qrxdedj1x8km0ksmc8fwq",
    "Xiangling-0000_j816336nczz00zb3kqzxxnuve3ub5w2",
    "Xiaogong-0000_jqcxe1bc0knjaxxoj5ztkyewbck49l7",
    "Xinyan-0000_m9dmptn3n55dm3f2usz49fnno916knc",
    "Yanfei-0000_1uib59lusm39lajs3huhxoyb4hc9l30",
    "Yelan-0000_kthng1wgpyyj8he68tx8cp87791wv5l",
    "Youla-0000_qdukmlynp0obxk2qs7h7yhzz5v86n6u",

    "Alexstrasza-IntroResponseLife00_000068",
    "Ana-Ultimate1Used00_000404",
    "Chromie-Kill01ALT1_000457",
    "DVa-UILockin00_000641",
    "Jaina-Taunt01_000799",
    "LieKong-UILockin00_000982",
    "LuNaLa-AIGoodJob02_000996",
    "MaWei-Pissed03_001296",

    "aishell3_SSB1956-SSB19560478_063258",
    "aishell3_SSB1341-SSB13410113_053277",
    "aishell3_SSB1956-SSB19560475_063255",
    "aishell3_SSB1575-SSB15750398_055947",
    "aishell3_SSB1448-SSB14480349_054146",
]


def split_train_test(ori_label_file: str):
    """ 输入 label文件、划分训练集、测试集； """

    # 写入：训练集
    train_file = ori_label_file.replace(".txt", ".train.txt")
    test_file = ori_label_file.replace(".txt", ".test.txt")

    # 正式运行：
    test_cnt = 0
    train_cnt = 0
    with open(ori_label_file, 'r', encoding='utf-8') as r1, \
            open(train_file, 'w', encoding='utf-8') as w1, \
            open(test_file, 'w', encoding='utf-8') as w2:

        w1.write(" ".join(["spk", "duration", "text", "path"]) + "\n")
        w2.write(" ".join(["spk", "duration", "text", "path"]) + "\n")

        lines = r1.readlines()

        for line in tqdm.tqdm(lines):
            if line.startswith("spk"):  # 去掉首行
                continue

            try:
                spk, dur, text, wav_ori_path = line.strip().split(" ", 3)
                basename = os.path.basename(wav_ori_path.replace(".wav", ""))
            except:
                print(f"ignore split error line: {line.strip()}")
                continue

            find = False
            for basename_i in test_basename_sets:
                if basename_i in basename:
                    w2.write(" ".join([spk, dur, text, wav_ori_path]) + "\n")
                    find = True
                    test_cnt += 1
                    break
            if find is False:
                w1.write(" ".join([spk, dur, text, wav_ori_path]) + "\n")
                train_cnt += 1

    print(f"Totally write {test_cnt} test data and {train_cnt} train data.")

    return


if __name__ == "__main__":
    split_train_test(
        ori_label_file="",
    )
