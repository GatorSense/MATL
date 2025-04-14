'''
This module contains the functions used for preprocessing of AWIR data.


Date created: April 14, 2025
Author:  Meilun Zhou

'''

import pandas as pd
import random
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import numpy as np

def extract_bounding_boxes_from_txt(file_path):
    with open(file_path, 'r') as f:
        # Each line in the text file corresponds to a bounding box
        boxes = [line.strip().split()[1:] for line in f.readlines()]  # Skip the class_id and get bounding box values
        return boxes

def create_dataframe(image_folder, image_extension, thermal=False, obj_class = 'cow'):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(image_extension)]
    
    data = []
    for image_file in image_files:
        # Check for corresponding text file (remove image extension and add .txt)
        if thermal:
            txt_file = image_file.replace(image_extension[2:], '.txt')
            #print(txt_file)
            
        else:
            txt_file = image_file.replace(image_extension, '.txt')
            
        txt_file_path = os.path.join(image_folder, txt_file)

        if os.path.exists(txt_file_path):
            bounding_boxes = extract_bounding_boxes_from_txt(txt_file_path)
            
            # For each bounding box in the text file, add a new row in the DataFrame
            for box in bounding_boxes:
                x_center, y_center, width, height = map(float, box)
                data.append([image_file, x_center, y_center, width, height, obj_class])
    
    # Create a DataFrame
    df = pd.DataFrame(data, columns=['image_name', 'x_center', 'y_center', 'width', 'height', 'class'])
    
    return df

# Function to search for an image file in the directory and its subdirectories
def find_image_path(image_folder, image_name):
    for root, dirs, files in os.walk(image_folder):
        if image_name in files:
            return os.path.join(root, image_name)
    return None

def plot_random_images_with_bboxes(df, image_folder, num_images=5):
    # Get 5 random samples from the DataFrame
    random_samples = df.sample(n=num_images)
    
    # Set up the figure with subplots
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    
    for i, (index, row) in enumerate(random_samples.iterrows()):
        image_name = row['image_name']
        x_center = row['x_center']
        y_center = row['y_center']
        width = row['width']
        height = row['height']
        class_label = row['class']
        
        # Search for the image in the folder and its subdirectories
        img_path = find_image_path(image_folder, image_name)
        if img_path and os.path.exists(img_path):
            # Load the image
            image = Image.open(img_path)
            axes[i].imshow(image)
            
            # Calculate bounding box coordinates
            img_width, img_height = image.size
            x_min = (x_center - width / 2) * img_width
            y_min = (y_center - height / 2) * img_height
            x_max = (x_center + width / 2) * img_width
            y_max = (y_center + height / 2) * img_height
            
            # Draw the bounding box
            rect = plt.Rectangle((x_min, y_min), width * img_width, height * img_height, edgecolor='red', facecolor='none', lw=2)
            axes[i].add_patch(rect)
            
            # Set title with the class label
            axes[i].set_title(f'Class: {class_label}')
            axes[i].axis('off')  # Turn off axis
        else:
            axes[i].set_title('Image Not Found')
            axes[i].axis('off')
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
def plot_class_images_with_bboxes(df, image_folder, animal_class, num_images=5):
    # Filter the DataFrame based on the specified class
    filtered_df = df[df['class'] == animal_class]
    
    if filtered_df.empty:
        print(f"No images found for class: {animal_class}")
        return
    
    # Get random samples from the filtered DataFrame
    random_samples = filtered_df.sample(n=min(num_images, len(filtered_df)))
    
    # Set up the figure with subplots
    fig, axes = plt.subplots(1, len(random_samples), figsize=(15, 5))
    
    for i, (index, row) in enumerate(random_samples.iterrows()):
        image_name = row['image_name']
        x_center = row['x_center']
        y_center = row['y_center']
        width = row['width']
        height = row['height']
        class_label = row['class']
        
        # Search for the image in the folder and its subdirectories
        img_path = find_image_path(image_folder, image_name)
        if img_path and os.path.exists(img_path):
            # Load the image
            image = Image.open(img_path)
            axes[i].imshow(image)
            
            # Calculate bounding box coordinates
            img_width, img_height = image.size
            x_min = (x_center - width / 2) * img_width
            y_min = (y_center - height / 2) * img_height
            x_max = (x_center + width / 2) * img_width
            y_max = (y_center + height / 2) * img_height
            
            # Draw the bounding box
            rect = plt.Rectangle((x_min, y_min), width * img_width, height * img_height, edgecolor='red', facecolor='none', lw=2)
            axes[i].add_patch(rect)
            
            # Set title with the class label
            axes[i].set_title(f'Class: {class_label}')
            axes[i].axis('off')  # Turn off axis
        else:
            axes[i].set_title('Image Not Found')
            axes[i].axis('off')
    
    # Show the plot
    plt.tight_layout()
    plt.show()

    # Function to filter images by size (200x200) and update filtered_df
def filter_images_and_df_by_size(df, rgb_images, thermal_images, target_size=(200, 200)):
    filtered_rgb = []
    filtered_thermal = []
    valid_indices = []

    # Iterate over the images and check their sizes
    for idx, (rgb_img, thermal_img) in enumerate(zip(rgb_images, thermal_images)):
        if rgb_img.shape[:2] == target_size and thermal_img.shape[:2] == target_size:
            filtered_rgb.append(rgb_img)
            filtered_thermal.append(thermal_img)
            valid_indices.append(idx)

    # Filter the DataFrame using the valid indices
    filtered_df = df.iloc[valid_indices].reset_index(drop=True)

    return filtered_rgb, filtered_thermal, filtered_df

def crop_random_placement(df, img_dir_rgb, img_dir_thermal, crop_size):
    rgb_crops = []
    thermal_crops = []
    new_x_centers = []
    new_y_centers = []
    new_widths = []
    new_heights = []
    valid_indices = []

    for idx, row in df.iterrows():
        # Get image names for both RGB and thermal
        rgb_image_name = row['rgb_image_name']
        thermal_image_name = row['thermal_image_name']

        # Load RGB and thermal images
        rgb_img_path = find_image_path(img_dir_rgb, rgb_image_name)
        thermal_img_path = find_image_path(img_dir_thermal, thermal_image_name)
        rgb_img = cv2.imread(rgb_img_path)
        thermal_img = cv2.imread(thermal_img_path)

        img_h, img_w, _ = rgb_img.shape
        
        # Convert normalized values to pixel coordinates
        x_center = int(row['x_center'] * img_w)
        y_center = int(row['y_center'] * img_h)
        bbox_width = int(row['width'] * img_w)
        bbox_height = int(row['height'] * img_h)

        # Calculate bounding box coordinates
        x_min = x_center - bbox_width // 2
        y_min = y_center - bbox_height // 2
        x_max = x_center + bbox_width // 2
        y_max = y_center + bbox_height // 2

        # Generate random offsets for object placement in crop (same for both RGB and thermal)
        max_offset_x = crop_size - bbox_width
        max_offset_y = crop_size - bbox_height
        
        offset_x = random.randint(0, max(0, max_offset_x))
        offset_y = random.randint(0, max(0, max_offset_y))

        # Ensure that the crop stays within the image boundaries for both images
        crop_x_min = max(0, x_min - offset_x)
        crop_y_min = max(0, y_min - offset_y)
        crop_x_max = min(img_w, crop_x_min + crop_size)
        crop_y_max = min(img_h, crop_y_min + crop_size)

        # Skip crops that exceed image boundaries
        if crop_x_min < 0 or crop_y_min < 0 or crop_x_max > img_w or crop_y_max > img_h:
            continue
            
        # Check if any other bounding boxes within the same image intersect with the crop
        crop_valid = True
        same_image_df = df[df['rgb_image_name'] == rgb_image_name]  # Only check rows with the same RGB image

        for other_idx, other_row in same_image_df.iterrows():
            if other_idx == idx:
                continue  # Skip checking against the current row

            # Convert other bounding box to pixel coordinates
            other_x_center = int(other_row['x_center'] * img_w)
            other_y_center = int(other_row['y_center'] * img_h)
            other_bbox_width = int(other_row['width'] * img_w)
            other_bbox_height = int(other_row['height'] * img_h)
            
            # Calculate other bounding box coordinates
            other_x_min = other_x_center - other_bbox_width // 2
            other_y_min = other_y_center - other_bbox_height // 2
            other_x_max = other_x_center + other_bbox_width // 2
            other_y_max = other_y_center + other_bbox_height // 2
            
            # Check if the other bounding box intersects the current crop
            if not (other_x_max < crop_x_min or other_x_min > crop_x_max or 
                    other_y_max < crop_y_min or other_y_min > crop_y_max):
                crop_valid = False
                break
        
        if not crop_valid:
            #print('another object in image')
            continue  # Skip the crop if any other bounding boxes intersect

        # Perform cropping for both RGB and thermal images
        rgb_cropped_img = rgb_img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        thermal_cropped_img = thermal_img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        # Append cropped images to the respective lists
        rgb_crops.append(rgb_cropped_img)
        thermal_crops.append(thermal_cropped_img)
        valid_indices.append(idx)  # Track the index of valid crops

        # Recalculate the normalized x_center, y_center, width, and height for the cropped image
        new_bbox_width = bbox_width / crop_size
        new_bbox_height = bbox_height / crop_size
        new_x_center = (x_center - crop_x_min) / crop_size
        new_y_center = (y_center - crop_y_min) / crop_size

        # Append the recalculated values to the respective lists
        new_x_centers.append(new_x_center)
        new_y_centers.append(new_y_center)
        new_widths.append(new_bbox_width)
        new_heights.append(new_bbox_height)

    # Filter df to keep only the valid rows
    filtered_df = df.iloc[valid_indices].copy()

    # Add new columns to the filtered DataFrame with the adjusted values
    filtered_df['new_x_center'] = new_x_centers
    filtered_df['new_y_center'] = new_y_centers
    filtered_df['new_width'] = new_widths
    filtered_df['new_height'] = new_heights

    return rgb_crops, thermal_crops, filtered_df

def plot_images_by_class(df, rgb_images, thermal_images, target_class, n_examples=1):
    # Filter the DataFrame by the specified class
    class_df = df[df['class'] == target_class]
    
    # Get the indices of the filtered images
    class_indices = class_df.index.tolist()
    
    # Select N random indices from the filtered images
    if len(class_indices) == 0:
        print(f"No images found for class '{target_class}'")
        return
    
    random_indices = random.sample(class_indices, min(n_examples, len(class_indices)))
    
    # Plot the selected RGB and thermal images
    plt.figure(figsize=(15, 5))
    
    for i, idx in enumerate(random_indices):
        # Get the bounding box coordinates for this image
        x_center = class_df.loc[idx, 'new_x_center']
        y_center = class_df.loc[idx, 'new_y_center']
        width = class_df.loc[idx, 'new_width']
        height = class_df.loc[idx, 'new_height']

        # Convert normalized bbox values to pixel coordinates
        img_h, img_w, _ = rgb_images[idx].shape
        bbox_width = int(width * img_w)
        bbox_height = int(height * img_h)
        x_min = int((x_center * img_w) - bbox_width / 2)
        y_min = int((y_center * img_h) - bbox_height / 2)
        x_max = x_min + bbox_width
        y_max = y_min + bbox_height

        # Get the image names for the titles
        rgb_image_name = class_df.loc[idx, 'rgb_image_name']
        thermal_image_name = class_df.loc[idx, 'thermal_image_name']

        # Plot the RGB image with bounding box
        plt.subplot(2, n_examples, i + 1)
        rgb_img_with_box = rgb_images[idx].copy()
        cv2.rectangle(rgb_img_with_box, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green bounding box
        plt.imshow(cv2.cvtColor(rgb_img_with_box, cv2.COLOR_BGR2RGB))
        plt.title(f'Index: {idx}')
        plt.axis('off')

        # Plot the thermal image with bounding box
        plt.subplot(2, n_examples, i + 1 + n_examples)
        thermal_img_with_box = thermal_images[idx].copy()
        cv2.rectangle(thermal_img_with_box, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green bounding box
        plt.imshow(cv2.cvtColor(thermal_img_with_box, cv2.COLOR_BGR2RGB))
        plt.title(f'Index: {idx}')
        plt.axis('off')

    plt.show()
    
def remove_index(N, filtered_df, filtered_rgb_cropped_images, filtered_thermal_cropped_images):
    # Check if the index exists
    if N not in filtered_df.index:
        print(f"Index {N} not found in DataFrame.")
        return filtered_df, filtered_rgb_cropped_images, filtered_thermal_cropped_images

    # Remove the row with index N from the DataFrame
    filtered_df = filtered_df.drop(index=N)

    # Remove the corresponding images from the rgb and thermal cropped image lists
    filtered_rgb_cropped_images.pop(N)
    filtered_thermal_cropped_images.pop(N)
    
    # Reset the index of the DataFrame
    filtered_df = filtered_df.reset_index(drop=True)

    return filtered_df, filtered_rgb_cropped_images, filtered_thermal_cropped_images

def convert_to_pixel_coordinates(df, img_width=300, img_height=300):
    # Initialize empty lists to store the coordinates
    xmins, xmaxs, ymins, ymaxs = [], [], [], []

    for idx, row in df.iterrows():
        # Convert normalized values to pixel coordinates
        x_center = row['new_x_center'] * img_width
        y_center = row['new_y_center'] * img_height
        width = row['new_width'] * img_width
        height = row['new_height'] * img_height

        # Calculate xmin, xmax, ymin, ymax
        xmin = x_center - (width / 2)
        xmax = x_center + (width / 2)
        ymin = y_center - (height / 2)
        ymax = y_center + (height / 2)

        # Append the coordinates to the respective lists
        xmins.append(int(xmin))
        xmaxs.append(int(xmax))
        ymins.append(int(ymin))
        ymaxs.append(int(ymax))

    # Add new columns for xmin, xmax, ymin, and ymax in the DataFrame
    df['xmin'] = xmins
    df['xmax'] = xmaxs
    df['ymin'] = ymins
    df['ymax'] = ymaxs

    return df