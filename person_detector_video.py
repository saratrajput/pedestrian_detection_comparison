import numpy as np
import cv2
import imutils

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#cv2.startWindowThread()

# open webcam video stream
cap = cv2.VideoCapture('vid.mp4')

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_per_second = float(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi', fourcc, 1, (frame_height, frame_width))
#out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
out = cv2.VideoWriter('output.avi', fourcc, frames_per_second, (int(cap.get(3)), int(cap.get(4))))

## the output will be written to output.avi
#out = cv2.VideoWriter(
#    'output.avi',
#    cv2.VideoWriter_fourcc(*'MJPG'),
#    15.,
#    (640,480))

while cap.isOpened():
    # capture frame-by-frame
    ret, image = cap.read()

    if ret:
        #image = imutils.resize(image,
        #        width=min(400, image.shape[1]))

        # Detecting all the regions in the image that has pedestrians in it
        (regions, _) = hog.detectMultiScale(image,
                                            winStride=(4,4),
                                            padding=(4,4),
                                            scale=1.05)

        # Drawing the regions in the Image
        for (x, y, w, h) in regions:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # showing the output image
        #image_write = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        #out.write(image_write.astype('uint8'))
        out.write(image.astype('uint8'))
        cv2.imshow("Image", image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
