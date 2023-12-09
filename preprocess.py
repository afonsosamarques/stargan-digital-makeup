import argparse
import os

from PIL import Image
from scipy.misc import imread
from torchvision import transforms as tr 


def rename_images(prev_dir, new_dir, wanted_length=5):
    for i, img in enumerate(os.listdir(prev_dir)):
        if img.endswith(".jpg") or img.endswith(".jpeg"):
            os_path = os.path.join(prev_dir, img)
            image = Image.fromarray(imread(os_path))
            new_filename = '0000' + str(i+1350)
            excess_size = len(new_filename) - wanted_length
            new_filename = new_filename[excess_size:] + ".jpg"
            image.save(os.path.join(new_dir, new_filename))
        elif img.endswith(".png"):
            os_path = os.path.join(prev_dir, img)
            image = Image.fromarray(imread(os_path))
            image = image.convert('RGB')
            new_filename = '0000' + str(i+1350)
            excess_size = len(new_filename) - wanted_length
            new_filename = new_filename[excess_size:] + ".jpg"
            image.save(os.path.join(new_dir, new_filename))


def rotate_images(lower_bound, upper_bound, image_dir):
    transform = []
    transform.append(tr.RandomAffine((lower_bound, upper_bound), resample=Image.BICUBIC))
    transform = tr.Compose(transform)
    for jpg in os.listdir(image_dir):
        if jpg.endswith(".jpg"):
            os_path = os.path.join(image_dir, jpg)
            image = Image.fromarray(imread(os_path))
            image = transform(image)
            image.save(os_path)


def adj_img_brightness(adjust_factor, image_dir):
    for jpg in os.listdir(image_dir):
        if jpg.endswith(".jpg"):
            os_path = os.path.join(image_dir, jpg)
            image = Image.fromarray(imread(os_path))
            image = tr.functional.adjust_brightness(image, adjust_factor)
            image.save(os_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_dir', type=str, default='/home/digital-makeup/makeup_pre')
    parser.add_argument('--write_dir', type=str, default='/home/digital-makeup/makeup_post')
    parser.add_arguement('--max_rotation', type=int, default=0, helper='Max rotation in degrees.')
    parser.add_argument('--brightness_adjust_factor', type=float, default=0.2, helper='Max brightness adjustment factor.')
    args = parser.parse_args()

    rename_images(args.read_dir, args.write_dir)
    rotate_images(-args.max_rotation, args.max_rotation, args.write_dir)
    adj_img_brightness(args.brightness_adjust_factor, args.write_dir)
