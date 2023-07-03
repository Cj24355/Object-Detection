import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import urllib.request
import numpy as np
from cvlib.object_detection import draw_bbox
import concurrent.futures
import time

url = "http://192.168.1.32/cam-hi.jpg"
im = None
metalCount = 0


def run1():
    cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)
    while True:
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        im = cv2.imdecode(imgnp, -1)

        cv2.imshow("live transmission", im)
        key = cv2.waitKey(5)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


def run2():
    global metalCount  # Declare as a global variable
    cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)
    while True:
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        im = cv2.imdecode(imgnp, -1)

        # Detect common objects in the image
        bbox, label, conf = cv.detect_common_objects(im)

        # Filter out objects that are not metal
        metal_indexes = [i for i, obj_label in enumerate(label) if obj_label == "metal"]
        metal_conf = [conf[i] for i in metal_indexes]
        metalCount = len(metal_indexes)  # Update the metal count

        if len(metal_indexes) > 0:
            # Draw bounding boxes and labels only for metal objects
            metal_bbox = [bbox[i] for i in metal_indexes]
            im = draw_bbox(im, metal_bbox, ["metal"] * len(metal_bbox), metal_conf)

        cv2.imshow("detection", im)
        key = cv2.waitKey(5)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


def countMetalObjects():
    global metalCount
    print("Number of metal objects detected:", metalCount)


def loop():
    while True:
        countMetalObjects()
        time.sleep(0.1)


if __name__ == "__main__":
    print("started")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        f1 = executor.submit(run1)
        f2 = executor.submit(run2)

    loop()  # Start the loop
