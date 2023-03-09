import os
from PIL import Image
import random
import imghdr
from math import radians
import sqlite3

# folder location containing all images
FOLDER_DIR = r"data/birds"
ROTATED_FOLDER_DIR = r"data/rotated_birds"


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
        img = img.resize((224, 224), Image.NEAREST)

        degrees_rotated = random.randint(0, 359)
        img = img.rotate(degrees_rotated)
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
