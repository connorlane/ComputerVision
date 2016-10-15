import cv2
import numpy as np

def detectHarrisCorners(image, gaussianSigma, neighborhoodSize, nmsRadius, desiredNumberOfCorners):
    # Define constants
    K = 0.05
    NMS_GAUSSIAN_BLUR_SIGMA = 2
    NMS_GAUSSIAN_BLUR_SIZE = 5

    # Pre-Calculate critical parameters 
    gaussianBlurSize = int((gaussianSigma*3)/2)*2 + 1 # Make kernel size 3 times the standard deviation -> 99.7%

    # Convert to grayscale, 32-bit float
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.float32) / 255

    # Apply the gaussian blur
    Kx = cv2.getGaussianKernel(gaussianBlurSize, gaussianSigma)
    Ky = np.transpose(Kx)
    img = cv2.sepFilter2D(img, -1, Kx, Ky)

    # Calculate gradients
    Kx = np.array([[-1, 0, 1]])
    Ky = np.array([[1], [0], [-1]])
    Gx = cv2.filter2D(img, -1, Kx)
    Gy = cv2.filter2D(img, -1, Ky)

    # Calculate products of derivatives (Gx^2, GxGy, and Gy^2)
    Gx2 = np.square(Gx)
    GxGy = np.multiply(Gx, Gy)
    Gy2 = np.square(Gy)

    # Compute sums of products of derivatives
    Sx2 = cv2.boxFilter(Gx2, -1, (neighborhoodSize, neighborhoodSize), normalize=False)
    Sxy = cv2.boxFilter(GxGy, -1, (neighborhoodSize, neighborhoodSize), normalize=False)
    Sy2 = cv2.boxFilter(Gy2, -1, (neighborhoodSize, neighborhoodSize), normalize=False)

    # Calculate the Harris R score
    detH = np.multiply(Sx2, Sy2) - np.square(Sxy)
    traceH = Sx2 + Sy2
    R = detH - K*np.square(traceH)

    # Do another Gaussian Blur to filter out noise before NMS
    Kx = cv2.getGaussianKernel(NMS_GAUSSIAN_BLUR_SIZE, NMS_GAUSSIAN_BLUR_SIGMA)
    Ky = np.transpose(Kx)
    Rblur = cv2.sepFilter2D(R, -1, Kx, Ky)

    # Perform non-maximum suppression (NMS)
    corners = []
    while len(corners) < desiredNumberOfCorners:
        _, _, _, maxLoc = cv2.minMaxLoc(Rblur)
        corners.append(maxLoc)
        cv2.circle(Rblur, maxLoc, nmsRadius, (0, 0, 0), thickness=-1) # Suppres all pixels in NMS RADIUS


    # Return the image and corner locations
    return corners, R
    
