from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt

def main():
    images_root = Path("./data/images")
    images_name = images_root.glob("**/*.jpg")
    for image_name in images_name:
        im = Image.open(image_name)
        im = im.resize((im.size[0]//8, im.size[1] // 8))
        plt.imshow(im.crop((0, 0, 448,448)))
        plt.show()
        break

if __name__ == "__main__":
    main()
