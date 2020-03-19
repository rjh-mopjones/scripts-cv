from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2

def convolve(image, kernel, padding):
    # grab the spatial dimensions of the image, along with
    # the spatial dimensions of the kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    # allocate memory for the output image, taking care to
    # "pad" the borders of the input image so the spatial
    # size (i.e., width and height) are not reduced
    pad = (kW - 1) // 2
    if (padding == "constant"):
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
            cv2.BORDER_REPLICATE)
    if (padding == "zero"):
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
            cv2.BORDER_CONSTANT, value=0)

    output = np.zeros((iH, iW), dtype="float32")

    # loop over the input image, "sliding" the kernel across
    # each (x, y)-coordinate from left-to-right and top to
    # bottom
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            # extract the ROI of the image by extracting the
            # *center* region of the current (x, y)-coordinates
            # dimensions
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]


            # perform the actual convolution by taking the
            # element-wise multiplicate between the ROI and
            # the kernel, then summing the matrix
            k = (roi * kernel).sum()


            # store the convolved value in the output (x,y)-
            # coordinate of the output image
            output[y - pad, x - pad] = k

    # rescale the output image to be in the range [0, 255]
    output = output.astype("int")

    # return the output image
    return output

prewitt_x = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
prewitt_y = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])


sobel_x = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
sobel_y = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])

laplacian = np.array([[0,1,0],[1,-4,1],[0,1,0]])

image = np.array([[3,4,8,15,25,45,50,52],
                 [3,4,8,15,25,45,50,52],
                 [3,4,8,15,25,45,50,52],
                 [3,4,8,15,25,45,50,52],
                 [3,4,8,15,25,45,50,52],
                 [3,4,8,15,25,45,50,52],
                 [3,4,8,15,25,45,50,52],
                 [3,4,8,15,25,45,50,52]])
image = np.array([[1,2,3,4,5],
                  [1,2,3,4,5],
                  [1,2,3,4,5],
                  [1,2,3,4,5],
                  [1,2,3,4,5],])




padding = "zero"

h_x = laplacian
h_y = laplacian

print("padding used:-" + padding)

g_x = convolve(image, np.flip(h_x), padding)
g_y = convolve(image, np.flip(h_y), padding)
print("g_x = ")
print(g_x)
print()
print("g_y = ")
print(g_y)
print()
print("g = sqrt( g_x^2 + g_y^2)")
print(np.around((np.sqrt(np.square(g_x)+np.square(g_y))),decimals=3))
print()
print("theta = tan-1(g_y/g_x)")
print(np.arctan(g_y/g_x)*180/np.pi)


