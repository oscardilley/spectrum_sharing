""" image_to_video.py

Using opencv to generate videos from simulation frames."""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import subprocess
import os
import re

def extract_number(filename):
    """Extracts the number from a filename, assuming it's the last number before the extension."""
    match = re.search(r'(\d+)(?=\D*$)', filename)  # Extract last number in filename
    return int(match.group(1)) if match else float('inf')  # Default to inf if no number found


def create_video_from_images(input_folder, output_file, frame_duration=0.2, temp_file="temp_output.avi"):
    temp_file = input_folder + "/" + temp_file
    # Get all PNG files in the folder
    # image_files = sorted(Path(input_folder).glob('*.png'))
    image_files = sorted(Path(input_folder).glob("*.png"), key=lambda x: extract_number(x.stem))
    
    if not image_files:
        raise ValueError(f"No PNG files found in {input_folder}")
    
    # Read first image to get dimensions
    test_img = plt.imread(str(image_files[0]))
    
    # Convert RGBA to RGB if necessary
    if test_img.shape[-1] == 4:  # If image has alpha channel
        height, width = test_img.shape[:2]
    else:
        height, width = test_img.shape[:2]
    
    # Calculate FPS based on desired frame duration
    fps = int(1/frame_duration)
    
    # First create an AVI file with MJPG codec (widely supported)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(temp_file, fourcc, fps, (width, height))
    
    for id, img_path in enumerate(image_files):
        # Read image using matplotlib (handles various PNG formats better)
        print(f"Processing frame {id}")
        img = plt.imread(str(img_path))
        
        # Convert RGBA to RGB if necessary
        if img.shape[-1] == 4:
            # Create white background
            background = np.ones((height, width, 3), dtype=np.uint8) * 255
            # Extract alpha channel
            alpha = img[:, :, 3]
            # Extract RGB channels
            rgb = img[:, :, :3]
            # Blend with white background
            img = (rgb * alpha[..., np.newaxis] + 
                  background * (1 - alpha[..., np.newaxis]))
        
        # Ensure image is in correct format for OpenCV
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)
        
        # Convert to BGR (OpenCV format)
        if len(img.shape) == 3:  # Color image
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:  # Grayscale image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Write frame
        out.write(img)
    
    # Release video writer
    out.release()
    
    # Convert to H.264 MP4 using FFmpeg
    try:
        subprocess.run([
            'ffmpeg', '-i', temp_file,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-pix_fmt', 'yuv420p',
            '-profile:v', 'main',  # PowerPoint-friendly profile
            '-level', '4.0',       # Good balance of quality and compatibility
            '-movflags', '+faststart',  # Enables faster start in PowerPoint
            '-y',
            output_file
        ], check=True)
        print(f"Video created successfully: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during FFmpeg conversion: {e}")
    except FileNotFoundError:
        print("FFmpeg not found. Please install FFmpeg on your system.")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)

# Example usage
if __name__ == "__main__":
    input_folder = "/home/ubuntu/spectrum_sharing/Videos/Primary2"
    output_file = "/home/ubuntu/spectrum_sharing/Videos/Primary2_video.mp4"
    create_video_from_images(input_folder, output_file)