import cv2
import numpy as np
import argparse

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument("logo_path", help="path to the logo image")
parser.add_argument("target_image_path", help="path to the target image")
args = parser.parse_args()

# Load the logo image
logo = cv2.imread(args.logo_path)

# Load the target image
target_image = cv2.imread(args.target_image_path)

# Convert the images to grayscale
logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
target_image_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

# Match the logo template in the target image
result = cv2.matchTemplate(target_image_gray, logo_gray, cv2.TM_CCOEFF_NORMED)

# Define a threshold for the matching score
threshold = 0.8

# Find the locations where the matching score is above the threshold
locations = np.where(result > threshold)

# If any locations are found, the logo is present in the target image
if len(locations[0]) > 0:
    print("The logo is present in the target image")
else:
    print("The logo is not present in the target image")
