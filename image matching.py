import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_histogram(image):
    histogram, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    return histogram

def calculate_cdf(histogram):
    cdf = histogram.cumsum()
    cdf_normalized = cdf / cdf.max()  # Normalize the CDF to range [0, 1]
    return cdf_normalized

def create_mapping_function(source_cdf, reference_cdf):
    mapping = np.zeros(256, dtype=np.uint8)
    for src_value in range(256):
        ref_value = np.argmin(np.abs(reference_cdf - source_cdf[src_value]))
        mapping[src_value] = ref_value
    return mapping

def apply_mapping(source_image, mapping):
    matched_image = mapping[source_image]
    return matched_image
def histogram_matching(source_image, reference_image):
    source_histogram = calculate_histogram(source_image)
    reference_histogram = calculate_histogram(reference_image)
    source_cdf = calculate_cdf(source_histogram)
    reference_cdf = calculate_cdf(reference_histogram)
    mapping = create_mapping_function(source_cdf, reference_cdf)
    matched_image = apply_mapping(source_image, mapping)
    return matched_image
def plot_images(source_image, reference_image, matched_image):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(source_image, cmap='gray')
    plt.title('Source Image')
    plt.subplot(1, 3, 2)
    plt.imshow(reference_image, cmap='gray')
    plt.title('Reference Image')
    plt.subplot(1, 3, 3)
    plt.imshow(matched_image, cmap='gray')
    plt.title('Matched Image')
    plt.show()

def main():
    source_image = cv2.imread('/content/WhatsApp Image 2024-05-01 at 15.39.28_7059a71f.jpg', cv2.IMREAD_GRAYSCALE)
    reference_image = cv2.imread('/content/tiger.jpg', cv2.IMREAD_GRAYSCALE)
    matched_image = histogram_matching(source_image, reference_image)
    cv2.imwrite('matched_image.png', matched_image)
    plot_images(source_image, reference_image, matched_image)

if __name__ == "__main__":
    main()
