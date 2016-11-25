#!/usr/bin/env python

import cv2
import sys
import numpy as np
from detectHarrisCorners import detectHarrisCorners

X = 0
Y = 1

# Get a patch from the image with the specified center and size
#   Returns: the image patch as a numpy array with <size> columns and <size> rows
#            Note: if the requested patch doesn't fit within the image, an empty array is returned
def getPatch(image, center, size):
    height, width = image.shape

    patchLeft = center[X] - size/2
    patchTop = center[Y] - size/2
    patchRight = center[X] + size/2 
    patchBottom = center[Y] + size/2 

    if patchLeft < 0 or patchTop < 0 or patchRight > width or patchBottom > height:
        return []

    patch = image[patchTop : patchBottom, patchLeft : patchRight]
    return patch

# Simply normalizes the specified image. Returns the normalized image
def normalize(image):
    mean, stddev = cv2.meanStdDev(image)
    return (image - mean) / stddev

# Calculates the similarity between two patches. Currently uses the sum of the difference of the
#    squares of corresponding pixels in each image
def calculateSimilarity(patch1, patch2):
    return np.square(np.subtract(patch1, patch2)).sum()

#  Finds the best matches between points in the source image and points in the test image
#  Returns: dictionary with keys as points in the source image and values as the best corresponding
#           point matches in the test image
def findBestMatches(sourcePositions, sourceImage, testPositions, testImage, patchSize):
    bestScoreMap = dict()

    for sourcePosition in sourcePositions:
        sourcePatch = getPatch(sourceImage, sourcePosition, patchSize)

        # If the patch is not completely contained within the image, skip it
        if not len(sourcePatch):
            continue

        # Normalize the patch to account for trivial image differences
        sourcePatch = normalize(sourcePatch)

        bestScore = None
        bestPosition = None

        for testPosition in testPositions:
            testPatch = getPatch(testImage, testPosition, patchSize) 

            # If the patch is not completely contained within the image, skip it
            if not len(testPatch):
                continue

            testPatch = normalize(testPatch)
            
            score = calculateSimilarity(testPatch, sourcePatch)

            # Update the current best score
            if not bestScore:
                bestScore = score
                bestPosition = testPosition
            elif score < bestScore:
                bestScore = score
                bestPosition = testPosition

        # Store the best position
        bestScoreMap[sourcePosition] = bestPosition

    return bestScoreMap

# Main function to detect matches at key locations between two images.
# Returns: A dictionary containing good matching points between image1 and image2
def matchFeatures(image1, image2, features1, features2, detectionPatchSize):
    image1gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    image2gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    bestScoreMap = findBestMatches(corners1, image1gray, corners2, image2gray, detectionPatchSize)
    bestScoreMapReversed = findBestMatches(corners2, image2gray, corners1, image1gray, detectionPatchSize)

    matches = dict()

    # Find matches such that the best match for the point in image1 is the best match for the point in image2
    for key in bestScoreMap:
        bestScore = bestScoreMap[key]
        if bestScoreMapReversed[bestScore] == key:
            matches[key] = bestScore

    return matches

def findHomography_OpenCV(src_pts, dst_pts):
    src_np = np.asarray(src_pts)
    dst_np = np.asarray(dst_pts)
    M, mask = cv2.findHomography(src_np, dst_np)
    return M

def filterRANSAC(matches):
    # Repeat M times
        # Randomly sample four from matches

        # Estimate homography from the matches 

        # Find inlier set

        # If the number of inliers is enough
            # Refine N times
                # Estimate the homography based on inlier set
                # Find new inlier set

        # If number of inliers in refined inlier set is more than previous iteration, save

    # Save the inlier set

# Define command line argument constants & strings
NUM_ARGUMENTS = 9
USAGE_MESSAGE = "Usage: " + sys.argv[0] + " <inputImagePath1> <inputImagePath2> <outputImagePath> <gaussianSigma> <neighborhoodSize> <nmsRadius> <desiredNumberOfCorners> <detectionPatchSize>"

# Display usage message if the number of arguments is wrong
if len(sys.argv) != NUM_ARGUMENTS:
    print USAGE_MESSAGE
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
image1 = cv2.imread(inputImagePath1)
image2 = cv2.imread(inputImagePath2)

# Run the harris corner detection algorithm
corners1, R1 = detectHarrisCorners(image1, gaussianSigma, neighborhoodSize, nmsRadius, desiredNumberOfCorners)
corners2, R2 = detectHarrisCorners(image2, gaussianSigma, neighborhoodSize, nmsRadius, desiredNumberOfCorners)

# Find the best matches between the two images
matches = matchFeatures(image1, image2, corners1, corners2, detectionPatchSize)

# Filter matches using RANSAC
matches = filterRANSAC(matches)

# Construct an image for visualizing the matches
visWidth = image1.shape[1] + image2.shape[1]
visHeight = max(image1.shape[0], image2.shape[0])
visImage = np.zeros((visHeight, visWidth, 3), dtype=np.uint8)
visImage[:image1.shape[0], :image1.shape[1],:] = image1
visImage[:image2.shape[0], image1.shape[1]:,:] = image2

# Draw lines & circles for each match from image1 to image2
for key in matches:
    bestCenter = matches[key]
    visSource = key
    visDestination = (bestCenter[0] + image1.shape[1], bestCenter[1])

    cv2.line(visImage, visSource, visDestination, (255, 0, 0), 1)

    cv2.circle(visImage, visSource, 7, (0, 255, 0))
    cv2.circle(visImage, visDestination, 7, (0, 0, 255))

# Display the resulting image
cv2.imshow('visImage', visImage)
cv2.waitKey(0)

# Save the visulaization image
print "Saving the image..."
cv2.imwrite(outputImagePath, visImage)

# Save the resulting image

cv2.destroyAllWindows()

# Exit
print "Done! Exiting..."

