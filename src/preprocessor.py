import os
from PIL import Image
import random
import imghdr
from math import radians
import sqlite3
import torchvision.transforms as transforms
import math

# folder location containing all images
FOLDER_DIR = r"data/birds"
ROTATED_FOLDER_DIR = r"data/rotated_birds"
# BARS_FOLDER_DIR = r"data/bar_birds"


def crop_around_center(img, width, height):
    """
    Given an image of size wxh, crops it to the given width and height around center
    """
    # define a transform to crop the image at center
    transform = transforms.CenterCrop((height, width))
    # crop the image using above defined transform
    img = transform(img)
    return img


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return wr, hr


def rotate_images():
    """
    Randomly downscales images to (244 x 244) and rotates all images in directory
    """

    conn = create_connection()

    for filename in os.listdir(FOLDER_DIR):
        filepath = rf"{FOLDER_DIR}/{filename}"
        rotated_file_path = rf"{ROTATED_FOLDER_DIR}/{filename}"
        # bar_file_path = rf"{BARS_FOLDER_DIR}/{filename}"
        if imghdr.what(filepath) != "jpeg":
            continue

        img = Image.open(filepath)
        og_width = img.width
        og_height = img.height

        degrees_rotated = random.randint(0, 359)
        radians_rotated = radians(degrees_rotated)
        img = img.rotate(degrees_rotated, expand=True)

        largest_rect = largest_rotated_rect(og_width, og_height, radians_rotated)

        img = crop_around_center(img, largest_rect[0], largest_rect[1])
        img = img.resize((224, 224), Image.NEAREST)
        img.save(rotated_file_path)
        with conn:
            # rotation is counter clockwise -> converting to clockwise
            degrees_rotated_clockwise = (360 - degrees_rotated) % 360
            radians_rotated = radians(degrees_rotated_clockwise)
            label = random.randint(1, 10)
            if label == 1:
                values = (filename, radians_rotated, "test")
            elif label == 2:
                values = (filename, radians_rotated, "validation")
            else:
                values = (filename, radians_rotated, "training")

            add_to_db(conn, values)


def create_connection():
    """
    creates a connection to the rotated birds database
    """
    conn = None
    try:
        conn = sqlite3.connect("data/birds.db")
    except Exception as e:
        print(e)

    return conn


def add_to_db(conn, values):
    """
    adds values to the database
    """
    with conn:
        # Note make sure that all changes are written from DB Browser if it
        # is being used
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO birds_split (filename,radians_rotated,label) VALUES (?, ?, ?)",
                values,
            )
        except Exception as e:
            print(e)


if __name__ == "__main__":
    rotate_images()
