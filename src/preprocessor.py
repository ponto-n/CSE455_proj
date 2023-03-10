import os
from PIL import Image
import random
import imghdr
from math import radians
import sqlite3
import torchvision.transforms as transforms
import math

# folder location containing all images
FOLDER_DIR = r"data/test_birds"
ROTATED_FOLDER_DIR = r"data/rotated_birds"

def crop_around_center(image, width, height):
    """
    Given an image, crops it to the given width and height,
    around it's centre point
    """
  # Read the image
    img = Image.open(image)
    # define a transform to crop the image at center
    transform = transforms.CenterCrop(height, width)
    # crop the image using above defined transform
    img = transform(img)
    return img

def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.
    """
    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )

def rotate_images():
    """
    Randomly downscales images to (244 x 244) and rotates all images in directory
    """

    conn = create_connection()

    for filename in os.listdir(FOLDER_DIR):
        filepath = rf"{FOLDER_DIR}/{filename}"
        rotated_file_path = rf"{ROTATED_FOLDER_DIR}/{filename}"
        if imghdr.what(filepath) != "jpeg":
            continue

        img = Image.open(filepath)
        #img = img.resize((224, 224), Image.NEAREST)

        degrees_rotated = random.randint(0, 359)
        img = img.rotate(degrees_rotated)
        #img.save(rotated_file_path)
        largest_rotated_rect = largest_rotated_rect(img.width, img.height, degrees_rotated)
        img = crop_around_center(img, img.width, img.height)
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
