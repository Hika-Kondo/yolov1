from pathlib import Path
from PIL import Image

def main():
    image_root = Path("../RawData/images")
    image_paths = image_root.glob("**/*.jpg")
    h_max, w_max = 1e5, 1e5
    for image_path in image_paths:
        image_path = str(image_path)
        im = Image.open(image_path)
        h, w = im.size
        if h < h_max:
            h_max = h
        if w < w_max:
            w_max = w
        print("now image h = {}, w = {}, h_max = {} w_max = {}".format(h, w, h_max, w_max))

    image_paths = image_root.glob("**/*.jpg")
    for image_path in image_paths:
        im = Image.open(image_path)
        im = im.crop((0,0,h_max,w_max))
        im.save(image_path)
        print("{} is croped".format(image_path))

if __name__ == "__main__":
    main()
