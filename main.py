#!/usr/bin/env python

import cv2
import sys
from detectHarrisCorners import detectHarrisCorners
import random
import math

import matching
import homography

# Define command line argument constants & strings
NUM_ARGUMENTS = 9
USAGE_MESSAGE = "Usage: " + sys.argv[0] + " <inputImagePath1> <inputImagePath2> <outputImagePath> <gaussianSigma> <neighborhoodSize> <nmsRadius> <desiredNumberOfCorners> <detectionPatchSize>"

# Display usage message if the number of arguments is wrong
if len(sys.argv) != NUM_ARGUMENTS:
    print USAGE_MESSAGE
    print "Suggested command: " + sys.argv[0] + " img/door1.jpg img/door2.jpg output.jpg 1.2 10 10 200 10"
    sys.exit(-1)

# Parse the input files
inputImagePath1        = sys.argv[1]
inputImagePath2        = sys.argv[2]
outputImagePath        = sys.argv[3]
gaussianSigma          = float(sys.argv[4])
neighborhoodSize       = int(sys.argv[5])
nmsRadius              = int(sys.argv[6])
desiredNumberOfCorners = int(sys.argv[7])
detectionPatchSize     = int(sys.argv[8])

# Read the input image from file
print "Reading the input images..."
image1 = cv2.imread(inputImagePath1)
image2 = cv2.imread(inputImagePath2)

# Run the harris corner detection algorithm
print "Detecting features using Harris Corner Detection..."
corners1, R1 = detectHarrisCorners(image1, gaussianSigma, neighborhoodSize, nmsRadius, desiredNumberOfCorners)
corners2, R2 = detectHarrisCorners(image2, gaussianSigma, neighborhoodSize, nmsRadius, desiredNumberOfCorners)

# Find the best matches between the two images, display the matches
print "Matching features between the two images..."
matches = matching.matchFeatures(image1, image2, corners1, corners2, detectionPatchSize)
matching.displayMatches(image1, image2, matches, 'Matched Features (Before RANSAC Filtering)')

# Filter matches using RANSAC
print "Filtering the matches using RANSAC..."
filtered_src_pts, H = matching.filterRANSAC(matches, 5, 1000, 100)
matching.displayMatches(image1, image2, filtered_src_pts, 'Matched Features (After RANSAC Filtering)')

# Stitch the image using the homography matrix
print "Stitching the images together (this could take 30 seconds or more)..."
stitchedImage = homography.stitchImages(image1, image2, H)

#warpedImage = cv2.warpPerspective(image1, H, (x_size + 200, y_size + 200), dst=stitchedImage)
cv2.imshow('Stitched Image', stitchedImage)
cv2.waitKey(0)

# Save the visulaization image
print "Saving the stitched image..."
cv2.imwrite(outputImagePath, stitchedImage)

# Save the resulting image
cv2.destroyAllWindows()

# Exit
print "Done! Exiting..."

