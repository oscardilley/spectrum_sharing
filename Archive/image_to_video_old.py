""" image_to_video.py

Using opencv to generate videos from simulation frames."""

import cv2
import os

def create_video_from_images(image_folder, output_video, fps=5):
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print("No images found in the specified folder.")
        return
    
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_image.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Error reading image: {img_path}")
            continue
        # Convert RGBA (4 channels) to RGB (3 channels) if needed
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        video.write(frame)
    
    video.release()
    print(f"Video saved as {output_video}")

# Example usage
image_folder = "/home/ubuntu/spectrum_sharing/Videos/Cam1"  # Change to your folder path
output_video = "/home/ubuntu/spectrum_sharing/Videos/Cam1_video.mp4"  # Change output file name if needed
create_video_from_images(image_folder, output_video)