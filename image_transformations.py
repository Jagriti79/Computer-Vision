import cv2
import numpy as np
import matplotlib.pyplot as plt

def negative_transformation(image):
    return 255 - image

def log_transformation(image):
    c = 255 / np.log(256)  # c for normalization
    return np.uint8(c * np.log1p(image))

def gamma_transformation(image, gamma=1.0):
    return np.uint8(255 * (image / 255.0) ** gamma)

def histogram_equalization(image):
    return cv2.equalizeHist(image)

def apply_transformations(image_path):
    # Load the grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply the transformations
    negative_image = negative_transformation(image)
    log_image = log_transformation(image)
    gamma_image = gamma_transformation(image, gamma=2.5)
    hist_eq_image = histogram_equalization(image)

    # Display the images
    images = [image, negative_image, log_image, gamma_image, hist_eq_image]
    titles = ['Original Image', 'Negative Transformation', 'Log Transformation',
              'Gamma Transformation', 'Histogram Equalization']

    plt.figure(figsize=(8, 10))
    for i in range(len(images)):
        plt.subplot(3, 2, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('on')
    plt.show()


apply_transformations('image path')
