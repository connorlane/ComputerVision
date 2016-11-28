#!/usr/bin/env python

import cv2
import sys
import numpy as np
from detectHarrisCorners import detectHarrisCorners
import random
import math

X = 0
Y = 1

INLIER_DISTANCE_THRESHOLD = 50
REQUIRED_INLIERS = 30

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
    src_np = np.asarray(src_pts, dtype=np.float32)
    dst_np = np.asarray(dst_pts, dtype=np.float32)
    print src_np
    print dst_np
    M, mask = cv2.findHomography(src_np, dst_np)
    return M

def warpBackward(src_image, h, dst_image):
    # Find the four corners
    columns, rows, depth = src_image.shape
    corners = [[0], [0], [1]], [[rows - 1], [0], [1]], [[0], [columns - 1], [1]], [[rows - 1], [columns - 1], [1]]
    mapped_corners = np.dot(corners, h)
    print mapped_corners

    # Use the four corners to calculate the destination image size for the warped image
    x_min = 256
    y_min = 256
    x_max = 0
    y_max = 0
    for corner in corners:
        corner_warped = np.dot(h, corner)
        corner_warped = corner_warped / corner_warped[2]
        x = int(corner_warped[1])
        y = int(corner_warped[0])

        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        if y > y_max:
            y_max = y

    x_offset = min(x_min, 0)
    y_offset = min(y_min, 0)
    x_size = max(src_image.shape[0], x_max)
    y_size = max(src_image.shape[1], y_max)

    print "x_offset: ", x_offset
    print "y_offset: ", y_offset
    print "x_size: ", x_size
    print "y_size: ", y_size
    
    # Calculate the inverse of h for performing backward warping
    h_inv = np.linalg.inv(h)
    h_inv = h_inv / h_inv[2, 2]

    # Create a blank image of the necessary size
    imageWarped = np.zeros((x_size + 1, y_size + 1, 3), dtype=np.uint8)
    visImage[x_offset:x_offset+src_image.shape[0], y_offset:y_offset+src_image.shape[1],:] = src_image

    # For each x, y in the destination image
    for x in range(x_offset, x_min + x_size):
        for y in range(y_min, y_min + y_size):

            # Calculate the coordinates in the orignal image
            locOrig = np.dot(h_inv, [[y],[x],[1]])
            locOrig = locOrig / locOrig[2]

            # Round to nearest location
            x_orig = int(round(locOrig[1]))
            y_orig = int(round(locOrig[0]))

            # If out of bounds on the original image
            if x_orig >= src_image.shape[0] or y_orig >= src_image.shape[1] or x_orig < 0 or y_orig < 0:
                imageWarped[x - x_min, y - y_min] = 0
            else:
                imageWarped[x - x_min, y - y_min] = src_image[x_orig, y_orig]

    # Return the warped image
    return imageWarped

def filterRANSAC(matches, distance_threshold, required_inliers, refinement_iterations):
    bestNumberOfInliers = 0
    print "RequriedInliers: ", required_inliers
    while bestNumberOfInliers < required_inliers:
        # Randomly sample four from matches
        selectedKeys = random.sample(matches, 4)
        selectedValues = [matches[k] for k in selectedKeys]
        
        # Estimate homography from the matches 
        H = findHomography_OpenCV(selectedKeys, selectedValues)
        print H

        inlierKeys = []
        # Find inlier set
        for src_pt in matches:
            dst_pt = matches[src_pt]
            warped_dst_pt = np.dot(H, [[src_pt[0]], [src_pt[1]], [1]])
            warped_dst_pt = warped_dst_pt / warped_dst_pt[2]
            distanceEstimator = (abs(dst_pt[0] - warped_dst_pt[0]) + abs(dst_pt[1] - warped_dst_pt[1])) / 2
            if distanceEstimator < distance_threshold:
                inlierKeys.append(src_pt)

        if len(inlierKeys) > bestNumberOfInliers:
            bestNumberOfInliers = len(inlierKeys)

    print "Num Inliers: ", bestNumberOfInliers

    for _ in range(refinement_iterations):
        # Estimate the homography based on inlier set
        inlierValues = [matches[k] for k in inlierKeys]
        H = findHomography_OpenCV(inlierKeys, inlierValues)

        # Find new inlier set
        inlierKeys = []

        # Find inlier set
        for src_pt in matches:
            dst_pt = matches[src_pt]
            warped_dst_pt = np.dot(H, [[src_pt[0]], [src_pt[1]], [1]])
            warped_dst_pt = warped_dst_pt / warped_dst_pt[2]
            distanceEstimator = (abs(dst_pt[0] - warped_dst_pt[0]) + abs(dst_pt[1] - warped_dst_pt[1])) / 2
            if distanceEstimator < distance_threshold:
                inlierKeys.append(src_pt)

            print "Refined inlier keys: ", len(inlierKeys)

    # Return the inlier keys and homography matrix
    return inlierKeys, H

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

# Filter matches using RANSAC
filtered_src_pts, H = filterRANSAC(matches, 256 * 0.1, desiredNumberOfCorners * 0.10, 100)

# Construct an image for visualizing the matches
visWidth = image1.shape[1] + image2.shape[1]
visHeight = max(image1.shape[0], image2.shape[0])
visImage = np.zeros((visHeight, visWidth, 3), dtype=np.uint8)
visImage[:image1.shape[0], :image1.shape[1],:] = image1
visImage[:image2.shape[0], image1.shape[1]:,:] = image2

# Draw lines & circles for each match from image1 to image2
for key in filtered_src_pts:
    bestCenter = matches[key]
    visSource = key
    visDestination = (bestCenter[0] + image1.shape[1], bestCenter[1])

    cv2.line(visImage, visSource, visDestination, (255, 0, 0), 1)

    cv2.circle(visImage, visSource, 7, (0, 255, 0))
    cv2.circle(visImage, visDestination, 7, (0, 0, 255))

# Display the resulting image
cv2.imshow('Matches - Filtered', visImage)
cv2.waitKey(0)


print H
print filtered_src_pts

rows, columns, depth = image1.shape
corners = np.array([[0, columns - 1,        0, columns - 1],
                    [0,           0, rows - 1,    rows - 1],
                    [1,           1,        1,           1]])

#corners = np.transpose(corners)

print corners
print "Columns: ", columns
print "Rows: ", rows
mapped_corners = np.dot(H, corners)
mapped_corners.T[:] = [row / row[2] for row in mapped_corners.T]
mapped_corners = mapped_corners[:2].T
print mapped_corners
print mapped_corners[:, :1]
x_min = int(math.floor(min(mapped_corners[:, :1])))
y_min = int(math.floor(min(mapped_corners[:, 1:])))
x_max = int(math.ceil(max(mapped_corners[:, :1])))
y_max = int(math.ceil(max(mapped_corners[:, 1:])))
print "x_min: ", x_min
print "y_min: ", y_min
print "x_max: ", x_max
print "y_max: ", y_max

x_offset = -min(0, x_min)
x_size = max(x_max, columns - 1) + x_offset + 1
y_offset = -min(0, y_min)
y_size = max(y_max, rows - 1) + y_offset + 1

print "x_offset: ", x_offset
print "y_offset: ", y_offset
print "x_size: ", x_size
print "y_size: ", y_size

stitchedImage = np.zeros((y_size, x_size, 3), dtype=np.uint8)
stitchedImage[y_offset:rows + y_offset, x_offset:columns + x_offset] = image2

# Calculate the inverse of h for performing backward warping
h_inv = np.linalg.inv(H)
h_inv = h_inv / h_inv[2, 2]

# For each x, y in the destination image

for x in range(x_min, x_max - 1):
    for y in range(y_min, y_max - 1):

        # Calculate the coordinates in the orignal image
        locOrig = np.dot(h_inv, [[x],[y],[1]])
        locOrig = locOrig / locOrig[2]

        # Round to nearest location
        x_orig = int(round(locOrig[1]))
        y_orig = int(round(locOrig[0]))

        #print "x_orig: ", x_orig
        #print "y_orig: ", y_orig

        # If out of bounds on the original image
        if x_orig >= rows or y_orig >= columns or x_orig < 0 or y_orig < 0:
            #imageWarped[x + x_offset, y + y_offset] = 0
            #print (columns, rows)
            #print (x_orig, y_orig), " out of bounds"
            pass
        else:
            #print "Setting: " , (x, y)
            #print (columns, rows)
            #print (x_orig, y_orig), " out of bounds"
            #print (x_orig, y_orig), " in bounds -> ", (x + x_offset, y + y_offset)
            stitchedImage[y + y_offset, x + x_offset] = image1[x_orig, y_orig]

#warpedImage = cv2.warpPerspective(image1, H, (x_size + 200, y_size + 200), dst=stitchedImage)
cv2.imshow('Stitched Image', stitchedImage)

# Display warped image
warpedImage = cv2.warpPerspective(image1, H, (1000, 1000), dst=image2)
cv2.imshow('warpedImage', warpedImage)

cv2.waitKey(0)

# Save the visulaization image
print "Saving the image..."
cv2.imwrite(outputImagePath, visImage)

# Save the resulting image

cv2.destroyAllWindows()

# Exit
print "Done! Exiting..."

