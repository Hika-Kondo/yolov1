import numpy as np
from PIL import Image
import pandas as pd

import argparse
from pathlib import Path


def remove_JA(inp):
    return inp.replace("下2000um", "").replace("丁E000um", "").replace("改③", "").replace("丁", "")


def main(args):
    # 出力ディレクトリを作る
    Path(args.output).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output)

    # 画像をresizeして保存する
    img_dir = Path(args.image_source)
    img_dir = list(img_dir.glob("**/*.jpg"))
    min_h, min_w = 1e10, 1e10
    for im_path in img_dir:
        img = Image.open(im_path)
        h,w = img.size
        if h < min_h:
            min_h = h
        if w < min_w:
            min_w = w
    min_h = min_h // args.scale
    min_w = min_w // args.scale

    # img_dir = Path(args.image_source).glob("**/*.jpg")
    for im_path in img_dir:
        im_name = str(im_path).split("/")[-1]
        out_name = remove_JA(im_name)
        out_name = out_name.split(" ")[-1]
        img = Image.open(im_path)
        h,w = img.size
        img = img.resize((min_h, min_w), resample=Image.BICUBIC)
        img.save(str(output_dir / out_name))
        print("save im_name: {}\n to {}".format(im_name, output_dir/out_name))


    with open(args.annotation) as f:
        lines = f.readlines()

    dataframe_dict = {"filename": [], "x_min": [], "y_min": [], "width": [], "height": [], "class": []}

    for line in lines:
        tmp = line.split(",")
        file_name = remove_JA(tmp[0][1::])
        if file_name == "filename":
            continue
        x_min = tmp[6].split(":")[-1]
        y_min = tmp[7].split(":")[-1]
        width = tmp[8].split(":")[-1]
        height = tmp[9].split(":")[-1]
        x_min = int("".join(filter(str.isalnum, x_min))) // args.scale
        y_min = int("".join(filter(str.isalnum, y_min))) // args.scale
        width = int("".join(filter(str.isalnum, width))) // args.scale
        height = int("".join(filter(str.isalnum, height))) // args.scale
        ans_class = 1 if "OK" in tmp[-1] else 0
        dataframe_dict["filename"].append(file_name)
        dataframe_dict["x_min"].append(x_min)
        dataframe_dict["y_min"].append(y_min)
        dataframe_dict["width"].append(width)
        dataframe_dict["height"].append(height)
        dataframe_dict["class"].append(ans_class)

    annotation_df = pd.DataFrame.from_dict(dataframe_dict)
    annotation_df.to_csv(path_or_buf=args.out_ano_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess")
    parser.add_argument("--image_source", type=str, help="画像ファイルのディレクトリ")
    parser.add_argument("--scale", type=int, help="何分の1にするかを決める")
    parser.add_argument("--output", type=str, help="どこのディレクトリに保存するか")
    parser.add_argument("--annotation", type=str, help="annotation csv file")
    parser.add_argument("--out_ano_name", type=str, help="annotation csv file")
    args = parser.parse_args()
    main(args)
