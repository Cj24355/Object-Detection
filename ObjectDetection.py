import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import time
import requests
import urllib.request
import numpy as np

url = "http://192.168.1.32/cam-hi.jpg"
bottleDetected = False
detection_count = 0


def run2():
    global bottleDetected, detection_count

    while True:
        # Retrieve image from the URL
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

        # Update the bottle detection flag and count
        bottleDetected = len(bottle_indexes) > 0
        detection_count += 1

        if bottleDetected:
            # Draw bounding boxes and labels only for plastic bottles
            bottle_bbox = [bbox[i] for i in bottle_indexes]
            im = draw_bbox(
                im,
                bottle_bbox,
                ["bottle"] * len(bottle_bbox),
                [conf[i] for i in bottle_indexes],
            )

            response = requests.get("http://192.168.1.32/bottle-detected")
            if response.status_code == 200:
                time.sleep(2)
                print("Notification sent to ESP32")
        else:
            print("No plastic bottle detected")

        cv2.imshow("detection", im)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


def loop():
    global bottleDetected, detection_count
    while True:
        # Check for the bottle detection condition
        if bottleDetected:
            time.sleep(3)
            print("Bottle is detected")
            print("Impulse Rate:", detection_count / 3, "detections per second")
            bottleDetected = (
                False  # Reset the variable after displaying the impulse rate
            )
            detection_count = 0  # Reset the detection count
        else:
            print("No bottle detected")

        # Add a delay between iterations to control the loop speed
        time.sleep(0.1)


if __name__ == "__main__":
    print("Started")
    run2()  # Run the detection function

    loop()  # Start the loop
