#!/usr/bin/env python

import cv2
import sys
import numpy as np
from detectHarrisCorners import detectHarrisCorners

NUM_ARGUMENTS = 8
USAGE_MESSAGE = sys.argv[0] + " <inputImagePath> <outputImagePath> <rscoreImagePath> <gaussianSigma> <neighborhoodSize> <nmsRadius> <desiredNumberOfCorners>"

if len(sys.argv) != NUM_ARGUMENTS:
    print USAGE_MESSAGE
    sys.exit(-1)

# Parse the input files
inputImagePath         = sys.argv[1]
outputImagePath        = sys.argv[2]
rscoreImagePath        = sys.argv[3]
gaussianSigma          = float(sys.argv[4])
neighborhoodSize       = int(sys.argv[5])
nmsRadius              = int(sys.argv[6])
desiredNumberOfCorners = int(sys.argv[7])

# Read the input image from file
image = cv2.imread(inputImagePath)

# Run the harris corner detection algorithm
corners, R = detectHarrisCorners(image, gaussianSigma, neighborhoodSize, nmsRadius, desiredNumberOfCorners)

# Normalize the R-Score values for easier viewing
R = R - np.amin(R)
R = 255 * R / np.amax(R)
R = R.astype(np.uint8)

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
