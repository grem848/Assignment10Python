import imutils
import cv2 as cv
import argparse

# optional: add cli args for running file
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# printing width, height and depth of the image
image = cv.imread(args["image"])
(h, w, d) = image.shape
print("width={}, height={}, depth={}".format(w, h, d))

# showing the image
cv.imshow("Shapes!", image)
cv.waitKey(0)

# resizing and showing resized image
resized = imutils.resize(image, width=350)
cv.imshow("Shapes resized with imutils!", resized)
cv.waitKey(0)

# convert the image to grayscale
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow("Now in grayscale!", gray)
cv.waitKey(0)

# showing the edges found in the image
edged = cv.Canny(gray, 30, 150)
cv.imshow("Only edges!", edged)
cv.waitKey(0)

# threshold the image
thresh = cv.threshold(gray, 225, 255, cv.THRESH_BINARY_INV)[1]
cv.imshow("Thresholded for finding contours!", thresh)
cv.waitKey(0)

# find contours of the foreground objects in the image
contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
	cv.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
output = image.copy()
 
# loop over the contours
for c in contours:
	# draw each contour on the output image with a 3px thick purple
	# outline, then display the output contours one at a time
	cv.drawContours(output, [c], -1, (240, 0, 159), 3)
	cv.imshow("Contours! Press any key!", output)
	cv.waitKey(0)

