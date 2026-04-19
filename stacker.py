import argparse
from PIL import Image

# Set up argument parser
parser = argparse.ArgumentParser(description="Stack two images horizontally.")
parser.add_argument("image1", help="Path to the first image")
parser.add_argument("image2", help="Path to the second image")
parser.add_argument("output", help="Path for the output stacked image")
args = parser.parse_args()

# Load the images
img1 = Image.open(args.image1)
img2 = Image.open(args.image2)

# Resize images to the same height
height = max(img1.height, img2.height)
img1 = img1.resize((int(img1.width * height / img1.height), height))
img2 = img2.resize((int(img2.width * height / img2.height), height))

# Create new image with combined width
combined_width = img1.width + img2.width
new_image = Image.new("RGB", (combined_width, height))

# Paste images side by side
new_image.paste(img1, (0, 0))
new_image.paste(img2, (img1.width, 0))

# Save the final image
new_image.save(args.output)
print(f"Images stacked horizontally and saved as '{args.output}'")

