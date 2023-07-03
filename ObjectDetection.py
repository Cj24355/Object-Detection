import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import urllib.request
import numpy as np
from cvlib.object_detection import draw_bbox
import concurrent.futures
import time
import requests

url = "http://192.168.1.32/cam-hi.jpg"
im = None
bottleDetected = False


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
    global bottleDetected  # Declare as a global variable
    cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)
    while True:
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        im = cv2.imdecode(imgnp, -1)

        # Detect common objects in the image
        bbox, label, conf = cv.detect_common_objects(im)

        # Filter out objects that are not plastic bottles
        bottle_indexes = [
            i
            for i, obj_label in enumerate(label)
            if obj_label == "bottle" and conf[i] > 0.5
        ]

        if len(bottle_indexes) > 0:
            # Bottle detected
            bottleDetected = True

            # Draw bounding boxes and labels only for plastic bottles
            bottle_bbox = [bbox[i] for i in bottle_indexes]
            im = draw_bbox(
                im,
                bottle_bbox,
                ["bottle"] * len(bottle_bbox),
                [conf[i] for i in bottle_indexes],
            )

        cv2.imshow("detection", im)
        key = cv2.waitKey(5)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


def playBuzzer():
    # Turn on the buzzer
    print("Bottle is Detected")
    # Code for activating the buzzer

    # Send a notification to the ESP32
    response = requests.get("http://192.168.1.100/bottle-detected")
    if response.status_code == 200:
        print("Notification sent to ESP32")


def loop():
    global bottleDetected  # Declare as a global variable
    while True:
        # Check for the bottle detection condition
        if bottleDetected:
            print("Bottle is detected")
            playBuzzer()
            bottleDetected = False  # Reset the variable after playing the buzzer
        else:
            print("No bottle detected")

        # Add a delay between iterations to control the loop speed
        time.sleep(0.1)


def detectBottle():
    global bottleDetected
    if bottleDetected:
        print("Bottle detected")


if __name__ == "__main__":
    print("started")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        f1 = executor.submit(run1)
        f2 = executor.submit(run2)

    loop()  # Start the loop
