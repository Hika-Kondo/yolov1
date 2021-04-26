from PIL import Image
from pathlib import Path

def main():
    root = Path("../RawData/resize_images")
    images = root.glob("**/*.jpg")
    image = list(images)[0]
    print(image)
    image = Image.open(image)
    image = image.crop((0,0,448,448))
    image.save("test.jpg")

if __name__ == "__main__":
    main()
