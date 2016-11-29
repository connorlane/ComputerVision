import math
import numpy as np

# Find using SVD
def findHomography(src_pts, dst_pts):
    if len(src_pts) != len(dst_pts):
        return -1

    # Get the A matrix
    A = np.zeros(shape=(2*len(src_pts), 9), dtype=float)
    for i in range (0, len(src_pts)):
        A[2*i:2*i + 2, :] = np.array([[src_pts[i][0], src_pts[i][1], 1, 0, 0, 0, -src_pts[i][0]*dst_pts[i][0], -src_pts[i][1]*dst_pts[i][0], -dst_pts[i][0]], \
                      [0, 0, 0, src_pts[i][0], src_pts[i][1], 1, -src_pts[i][0]*dst_pts[i][1], -src_pts[i][1]*dst_pts[i][1], -dst_pts[i][1]]])

    # Get the b matrix
    b = np.zeros(shape=(2*len(src_pts), 1), dtype=float)
    for i in range (0, len(src_pts)):
        b[2*i:2*i+2] = np.array([[dst_pts[i][0]], [dst_pts[i][1]]])

    # Calcualte the SVD
    u, s, v = np.linalg.svd(A)

    #smallest = np.where(s == min(s))[0][0]
    h = v[8, :]
    h = h * 1 / h[8]

    # Reshape
    h = np.reshape(h, (3, 3))

    return h

def stitchImages(image1, image2, H):
    rows, columns, depth = image1.shape
    rows2, columns2, depth2 = image2.shape
    corners = np.array([[0, columns - 1,        0, columns - 1],
                        [0,           0, rows - 1,    rows - 1],
                        [1,           1,        1,           1]])

    mapped_corners = np.dot(H, corners)
    mapped_corners.T[:] = [row / row[2] for row in mapped_corners.T]
    mapped_corners = mapped_corners[:2].T

    x_min = int(math.floor(min(mapped_corners[:, :1])))
    y_min = int(math.floor(min(mapped_corners[:, 1:])))
    x_max = int(math.ceil(max(mapped_corners[:, :1])))
    y_max = int(math.ceil(max(mapped_corners[:, 1:])))

    x_offset = -min(0, x_min)
    x_size = max(x_max, columns2 - 1) + x_offset + 1
    y_offset = -min(0, y_min)
    y_size = max(y_max, rows2 - 1) + y_offset + 1

    stitchedImage = np.zeros((y_size, x_size, 3), dtype=np.uint8)
    stitchedImage[y_offset:rows2 + y_offset, x_offset:columns2 + x_offset] = image2

    # Calculate the inverse of h for performing backward warping
    h_inv = np.linalg.inv(H)
    h_inv = h_inv / h_inv[2, 2]

    # For each x, y in the destination image (backward warping)
    for x in range(x_min, x_max - 1):
        for y in range(y_min, y_max - 1):

            # Calculate the coordinates in the orignal image
            locOrig = np.dot(h_inv, [[x],[y],[1]])
            locOrig = locOrig / locOrig[2]

            # Round to nearest location
            x_orig = int(round(locOrig[1]))
            y_orig = int(round(locOrig[0]))

            # If in bounds on the original image
            if x_orig < rows and y_orig < columns and x_orig >= 0 and y_orig >= 0:
                stitchedImage[y + y_offset, x + x_offset] = image1[x_orig, y_orig]

    return stitchedImage


