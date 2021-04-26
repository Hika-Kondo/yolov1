from PIL import Image
from PIL import ImageDraw
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd


def show_bb(img, x, y, w, h, text="ans", textcolor=(255,255,255), bbcolor=(255,0,0)):
    draw = ImageDraw.Draw(img)
    text_w, text_h = draw.textsize(text)
    label_y = y if y <= text_h else y - text_h
    draw.rectangle((x, label_y, x+w, label_y+h), outline=bbcolor)
    draw.rectangle((x, label_y, x+text_w, label_y+text_h), outline=bbcolor, fill=bbcolor)
    draw.text((x, label_y), text, fill=textcolor)


def main():
    df = pd.read_csv("test/annotation.csv")
    img = Image.open("./test/0326-SR2-SH6T-0130-0230-1A.jpg")
    tmp = df[df["filename"] == "0326-SR2-SH6T-0130-0230-1A.jpg"]
    x = tmp["x_min"]; y = tmp["y_min"]; width = tmp["width"]; height = tmp["height"]

    # img = Image.open("../RawData/val/0326-SR2-SH6T-0130-0230-1A.jpg")
    # boxes = []
    # with open("../pytorch-YOLO-v1/bt_im.txt") as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         li_li = line.split(" ")
    #         if li_li[0] == "0326-SR2-SH6T-0130-0230-1A.jpg":
    #             li = li_li[1:-1]
    #             print(li)
    #             for i in range(0,len(li),5):
    #                 b = []
    #                 b.append(int(li[i]))
    #                 b.append(int(li[i+1]))
    #                 b.append(int(li[i+2]))
    #                 b.append(int(li[i+3]))
    #                 boxes.append(b)

    for idx in range(len(x)):
        show_bb(img, x[idx], y[idx], width[idx], height[idx], )
    img.save("res.png")


if __name__ == "__main__":
    main()
