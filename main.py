import cv2

# Load the Haar Cascade classifiers for detecting faces and eyes
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load the image from the specified path
img_path = 'image.png'
img = cv2.imread(img_path)

# Convert the image to grayscale for better face detection
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Perform face detection on the grayscale image
detected_faces = face_detector.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)

# Loop through all the detected faces and draw rectangles around them
for (x, y, w, h) in detected_faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Save the image with detected faces or show it in a window
cv2.imwrite('faces_detected_output.jpg', img)  # Save the image with rectangles drawn
cv2.imshow('Detected Faces', img)  # Display the image with rectangles drawn
cv2.waitKey(0)
cv2.destroyAllWindows()
