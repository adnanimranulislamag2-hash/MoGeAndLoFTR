import cv2
import torch
import numpy as np
from moge.model.v2 import MoGeModel
import os

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model from huggingface hub
print("Loading MoGe-2 model...")
model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
print("Model loaded successfully!")

# Input and output paths
input_path = "/root/moge_work/images/ test101.png"  # Change this to your image path
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Read the input image and convert to tensor (3, H, W) with RGB values normalized to [0, 1]
print(f"Reading image from {input_path}...")
input_image_bgr = cv2.imread(input_path)
if input_image_bgr is None:
    raise ValueError(f"Could not read image at {input_path}")

input_image_rgb = cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB)
input_image = torch.tensor(input_image_rgb / 255, dtype=torch.float32, device=device).permute(2, 0, 1)

print(f"Image shape: {input_image.shape}")

# Infer
print("Running inference...")
with torch.no_grad():  # Save memory
    output = model.infer(input_image)

print("Inference complete!")
print(f"Output keys: {output.keys()}")

# Save outputs
base_name = os.path.splitext(os.path.basename(input_path))[0]

# 1. Save depth map
depth = output["depth"].cpu().numpy()
depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
cv2.imwrite(f"{output_dir}/{base_name}_depth.png", depth_colored)
print(f"Saved depth map to {output_dir}/{base_name}_depth.png")

# 2. Save normal map
if "normal" in output:
    normal = output["normal"].cpu().numpy()
    # Convert from [-1, 1] to [0, 255]
    normal_vis = ((normal + 1) / 2 * 255).astype(np.uint8)
    normal_vis_bgr = cv2.cvtColor(normal_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{output_dir}/{base_name}_normal.png", normal_vis_bgr)
    print(f"Saved normal map to {output_dir}/{base_name}_normal.png")

# 3. Save mask
mask = output["mask"].cpu().numpy()
mask_vis = (mask * 255).astype(np.uint8)
cv2.imwrite(f"{output_dir}/{base_name}_mask.png", mask_vis)
print(f"Saved mask to {output_dir}/{base_name}_mask.png")

# 4. Save point cloud as PLY
points = output["points"].cpu().numpy()
mask_np = mask.astype(bool)

# Get valid points and colors
valid_points = points[mask_np]
valid_colors = input_image_rgb[mask_np]

# Write PLY file
ply_path = f"{output_dir}/{base_name}_pointcloud.ply"
with open(ply_path, 'w') as f:
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write(f"element vertex {len(valid_points)}\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write("end_header\n")
    for point, color in zip(valid_points, valid_colors):
        f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")

print(f"Saved point cloud to {ply_path}")

# 5. Print camera intrinsics
intrinsics = output["intrinsics"].cpu().numpy()
print("\nCamera Intrinsics:")
print(intrinsics)

print("\n✅ All outputs saved successfully!")
print(f"Point map shape: {output['points'].shape}")
print(f"Depth map shape: {output['depth'].shape}")
print(f"Normal map shape: {output['normal'].shape if 'normal' in output else 'N/A'}")