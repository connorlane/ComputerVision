import numpy as np
import cv2
import random

import homography

# Get a patch from the image with the specified center and size
#   Returns: the image patch as a numpy array with <size> columns and <size> rows
#            Note: if the requested patch doesn't fit within the image, an empty array is returned
def getPatch(image, center, size):
    height, width = image.shape

    patchLeft = center[0] - size/2
    patchTop = center[1] - size/2
    patchRight = center[0] + size/2 
    patchBottom = center[1] + size/2 

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
def matchFeatures(image1, image2, corners1, corners2, detectionPatchSize):
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

# Displays the two images side by side with connecting lines & circles
def displayMatches(image1, image2, matches, windowName):
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
    cv2.imshow(windowName, visImage)
    cv2.waitKey(0)

# Filters the matches based on a homography model using RANSAC algorithm
def filterRANSAC(matches, distance_threshold, inlier_iterations, refinement_iterations):
    bestNumberOfInliers = 0
    for _ in range(inlier_iterations):
        # Randomly sample four from matches
        selectedKeys = random.sample(matches, 4)
        selectedValues = [matches[k] for k in selectedKeys]
        
        # Estimate homography from the matches 
        H = homography.findHomography(selectedKeys, selectedValues)

        selectedInlierKeys = []
        # Find inlier set
        for src_pt in matches:
            dst_pt = matches[src_pt]
            warped_dst_pt = np.dot(H, [[src_pt[0]], [src_pt[1]], [1]])
            warped_dst_pt = warped_dst_pt / warped_dst_pt[2]
            distanceEstimator = (abs(dst_pt[0] - warped_dst_pt[0]) + abs(dst_pt[1] - warped_dst_pt[1])) / 2
            if distanceEstimator < distance_threshold:
                selectedInlierKeys.append(src_pt)

        if len(selectedInlierKeys) > bestNumberOfInliers:
            bestNumberOfInliers = len(selectedInlierKeys)
            inlierKeys = selectedInlierKeys

    for _ in range(refinement_iterations):
        # Estimate the homography based on inlier set
        inlierValues = [matches[k] for k in inlierKeys]
        H = homography.findHomography(inlierKeys, inlierValues)

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

    # Return the inlier keys and homography matrix
    filteredMatches = dict((k, matches[k]) for k in inlierKeys)
    return filteredMatches, H


