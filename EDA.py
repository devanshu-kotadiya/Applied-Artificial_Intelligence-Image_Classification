import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import skew
dataset_path = "Training"

classes = ["museum-indoor", "museum-outdoor"]
def dataset_overview(dataset_path,classes):
    image_counts = {}

    for class_label in classes:
        class_path = os.path.join(dataset_path, class_label)
    
        if not os.path.exists(class_path):
            print(f"Warning: {class_label} directory not found.")
            continue
    
        images = os.listdir(class_path)
        image_counts[class_label] = len(images)

    print("Dataset Overview:")
    for class_label, count in image_counts.items():
        print(f"  {class_label}: {count} images")

    print(f"\nTotal Images: {sum(image_counts.values())}")


dataset_overview(dataset_path,classes)

def display_sample_images(dataset_path, classes, num_samples=3):
    fig, axes = plt.subplots(len(classes), num_samples, figsize=(12, 6))
    
    for i, class_label in enumerate(classes):
        class_path = os.path.join(dataset_path, class_label)
        image_files = os.listdir(class_path)
        
        
        sample_images = random.sample(image_files, min(num_samples, len(image_files)))
        
        for j, image_file in enumerate(sample_images):
            image_path = os.path.join(class_path, image_file)
            img = Image.open(image_path)
            axes[i, j].imshow(img)
            axes[i, j].axis("off")
            axes[i, j].set_title(f"{class_label}")
    
    plt.tight_layout()
    plt.show()

def check_image_dimensions(dataset_path, classes):
    dimensions = []
    for class_label in classes:
        class_path = os.path.join(dataset_path, class_label)
        image_files = os.listdir(class_path)
        
        for image_file in image_files[:10]:
            image_path = os.path.join(class_path, image_file)
            img = Image.open(image_path)
            dimensions.append((img.width, img.height))
    
    dimensions = np.array(dimensions)
    
    print("\nImage Dimensions Statistics:")
    print(f"  Mean Width: {np.mean(dimensions[:, 0]):.2f}, Mean Height: {np.mean(dimensions[:, 1]):.2f}")
    print(f"  Min Width: {np.min(dimensions[:, 0])}, Max Width: {np.max(dimensions[:, 0])}")
    print(f"  Min Height: {np.min(dimensions[:, 1])}, Max Height: {np.max(dimensions[:, 1])}")

def plot_pixel_histograms(dataset_path, classes, num_samples=5):
    fig, axes = plt.subplots(len(classes), 1, figsize=(10, 5 * len(classes)))

    for i, class_label in enumerate(classes):
        class_path = os.path.join(dataset_path, class_label)
        image_files = os.listdir(class_path)
        sample_images = random.sample(image_files, min(num_samples, len(image_files)))

        pixel_values = []
        for image_file in sample_images:
            image_path = os.path.join(class_path, image_file)
            img = Image.open(image_path).convert("L") 
            pixel_values.extend(np.array(img).flatten())

       
        axes[i].hist(pixel_values, bins=50, color='gray', alpha=0.7)
        axes[i].set_title(f"Pixel Intensity Distribution: {class_label}")
        axes[i].set_xlabel("Pixel Value (0-255)")
        axes[i].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

display_sample_images(dataset_path, classes)
check_image_dimensions(dataset_path, classes)
plot_pixel_histograms(dataset_path, classes)

def compute_image_statistics(dataset_path, classes, num_samples=10):
    stats = {class_label: [] for class_label in classes}

    for class_label in classes:
        class_path = os.path.join(dataset_path, class_label)
        image_files = os.listdir(class_path)
        sample_images = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)

        for image_file in sample_images:
            image_path = os.path.join(class_path, image_file)
            img = Image.open(image_path).convert("L")  

            img_array = np.array(img).flatten()
            mean_value = np.mean(img_array)
            variance_value = np.var(img_array)
            skewness_value = skew(img_array)

            stats[class_label].append((mean_value, variance_value, skewness_value))

    
    for class_label in classes:
        stats[class_label] = np.array(stats[class_label])
        print(f"\nStatistics for {class_label}:")
        print(f"  Mean: {np.mean(stats[class_label][:, 0]):.2f}")
        print(f"  Variance: {np.mean(stats[class_label][:, 1]):.2f}")
        print(f"  Skewness: {np.mean(stats[class_label][:, 2]):.2f}")

compute_image_statistics(dataset_path, classes)

def analyze_rgb_distribution(dataset_path, classes, num_samples=5):
    fig, axes = plt.subplots(len(classes), 3, figsize=(15, 5 * len(classes)))

    for i, class_label in enumerate(classes):
        class_path = os.path.join(dataset_path, class_label)
        image_files = os.listdir(class_path)
        sample_images = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)

        r_vals, g_vals, b_vals = [], [], []

        for image_file in sample_images:
            image_path = os.path.join(class_path, image_file)
            img = Image.open(image_path).convert("RGB")  # Ensure RGB format
            img_array = np.array(img)

            r_vals.extend(img_array[:, :, 0].flatten())
            g_vals.extend(img_array[:, :, 1].flatten())
            b_vals.extend(img_array[:, :, 2].flatten())

        # Plot histograms for each color channel
        axes[i, 0].hist(r_vals, bins=50, color='red', alpha=0.7)
        axes[i, 0].set_title(f"{class_label} - Red Channel")

        axes[i, 1].hist(g_vals, bins=50, color='green', alpha=0.7)
        axes[i, 1].set_title(f"{class_label} - Green Channel")

        axes[i, 2].hist(b_vals, bins=50, color='blue', alpha=0.7)
        axes[i, 2].set_title(f"{class_label} - Blue Channel")

    plt.tight_layout()
    plt.show()

analyze_rgb_distribution(dataset_path, classes)