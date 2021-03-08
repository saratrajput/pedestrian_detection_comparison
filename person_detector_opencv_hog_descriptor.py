import os
import numpy as np
import cv2
import imutils
import argparse
import logging
# pylint: disable=import-error, no-member, invalid-name

def main(args):
    # Get directory name of only last folder in path
    input_file = os.path.join(args.input_video)
    output_file = os.path.join(args.output_video)

    # Initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Load video file
    cap = cv2.VideoCapture(input_file)

    # Set the format for output file
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = float(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, frames_per_second, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        # capture frame-by-frame
        ret, image = cap.read()

        if ret:

            # Detecting all the regions in the image that has pedestrians in it
            (regions, _) = hog.detectMultiScale(image,
                                                winStride=(4,4),
                                                padding=(4,4),
                                                scale=1.05)

            # Drawing the regions in the Image
            for (x, y, w, h) in regions:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Save video
            out.write(image.astype('uint8'))
            # Display output image
            cv2.imshow("Image", image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()


    parser.add_argument("-i", "--input_video",
                        default='vid.mp4',
                        help="Input video file")

    parser.add_argument("-o", "--output_video",
                        default='output.avi',
                        help="Output video file")

    args = parser.parse_args()

    main(args)
