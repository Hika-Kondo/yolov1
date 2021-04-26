from PIL import Image
from pathlib import Path


def im_read_save(path):
        try:
            im = Image.open(path)
        except:
            return 0
        save_dir = str(images_dir / im_name)
        im.save(save_dir)
        print("{} is copied in {}".format(im_name, save_dir))

def main():
    data_dir = "../RawData"
    root = Path("./{}".format(data_dir))
    Path("./{}/images".format(data_dir)).mkdir(exist_ok=True, parents=True)
    images_dir = Path("./{}/images".format(data_dir))
    image_path = root.glob("**/*.jpg")
    for path in image_path:
        path_str = str(path)
        im_name = path_str.split("/")[-1]
        im_name = im_name.split()[-1]
        im_name = im_name.replace("下2000um","丁E000um")
        im_name = im_name.replace("改③", "").replace("丁", "")
        try:
            im = Image.open(path)
        except:
            continue
        save_dir = str(images_dir / im_name)
        im.save(save_dir)
        print("{} is copied in {}".format(im_name, save_dir))

if __name__ == "__main__":
    main()
