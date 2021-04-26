from PIL import Image
from pathlib import Path


def main():
    root = Path("../RawData")
    souce_dir = root / "images"
    save_dir = root / "resize_images"
    save_dir.mkdir(exist_ok=True, parents=True)
    souce_images = souce_dir.glob("**/*.jpg")
    for image_path in souce_images:
        fname = str(image_path).split("/")[-1]
        im = Image.open(image_path)
        h, w = im.size
        im = im.resize((h//8,w//8), resample=Image.BICUBIC)
        im.save(str(save_dir / fname))
        print("{} is resize and saved in {}".format(image_path, str(save_dir / fname)))
        print(h//8, w//8)

if __name__ == "__main__":
    main()
