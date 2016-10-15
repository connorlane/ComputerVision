#!/usr/bin/env python

import cv2
import sys
from detectHarrisCorners import detectHarrisCorners

NUM_ARGUMENTS = 4
USAGE_MESSAGE = sys.argv[0] + " <inputImagePath> <outputImagePath> <rscoreImagePath>"

if len(sys.argv) != NUM_ARGUMENTS:
    print USAGE_MESSAGE
    exit(-1)

# Parse the input files
inputImagePath = sys.argv[1]
outputImagePath = sys.argv[2]
rscoreImagePath = sys.argv[3]

# Read the input image from file
image = cv2.imread(inputImagePath)

# Run the harris corner detection algorithm
corners, R = detectHarrisCorners(image, 3, 7, 10, 100)

# Print the corner coordinates
print "Corner coordinates:"
for corner in corners:
    print corner

# Draw the corner locations on the original image
for c in corners:
    cv2.circle(image, c, 7, (0, 0, 255), thickness=1)

# Display the original image & R-Score image
cv2.imshow("Original image with corners", image)
cv2.imshow("Harris R-Score Image", R)
cv2.waitKey(0)

# Save the images to files
print "Saving images..."
cv2.imwrite(outputImagePath, image)
cv2.imwrite(rscoreImagePath, R)

# Exit
print "Done! Exiting..."
cv2.destroyAllWindows()
