import cv2

pedestrian_cascade = cv2.CascadeClassifier(r"xmls\haarcascade_frontalface_default.xml")

# Function to perform pedestrian detection from images. Pass an image as a variable.
def pedestrianDetection(frame):

    pedestrians = pedestrian_cascade.detectMultiScale(frame, 1.2, 5)
    # To draw a rectangle on each pedestrian
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, 'Person', (x + 6, y - 6), font, 0.5, (0, 255, 0), 1)
    # Display frames in a window
    return frame

#input image
gray = cv2.imread(r"files\face.jpeg")
#gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
output_image = pedestrianDetection(gray)

#resizing image
scale_percent = 50
resized_img = cv2.resize(output_image, (int(output_image.shape[1] * scale_percent / 100),
                                       int(output_image.shape[0] * scale_percent / 100)),
                                 interpolation=cv2.INTER_AREA)

cv2.imwrite('detected_face1.png', output_image)

# Display the output image
cv2.imshow("Pedestrian Detection", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
