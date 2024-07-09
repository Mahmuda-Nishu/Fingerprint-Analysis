import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the fingerprint image
image_path = 'C:\\Users\ManMis\\Downloads\\fingerprint.png'  # Replace with the path to your image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load image.")
    exit()

# Apply Gaussian Blur to reduce noise
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Apply edge detection (Canny)
edges = cv2.Canny(blurred_image, 50, 150)

# Detect key points using Harris corner detection
harris_corners = cv2.cornerHarris(blurred_image, 2, 3, 0.04)
keypoints = np.argwhere(harris_corners > 0.01 * harris_corners.max())

# Display the original and processed images
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(blurred_image, cmap='gray')
axes[1].set_title('Blurred Image')
axes[1].axis('off')

axes[2].imshow(edges, cmap='gray')
axes[2].set_title('Edges')
axes[2].axis('off')

plt.show()

# Analyze the image (basic statistics)
pixel_values = image.flatten()
mean_intensity = np.mean(pixel_values)
std_intensity = np.std(pixel_values)

print(f"Mean Intensity: {mean_intensity}")
print(f"Standard Deviation of Intensity: {std_intensity}")

# Visualize distribution of pixel values
plt.figure(figsize=(10, 5))
plt.hist(pixel_values, bins=50, color='gray')
plt.title('Histogram of Pixel Intensities')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()