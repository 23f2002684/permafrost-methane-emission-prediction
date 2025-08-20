import numpy as np
import cv2
import os

def physics_filter_batch(input_folder, output_file="risk_report.txt"):
    results = []
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".tif")):
            mask_path = os.path.join(input_folder, filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            
            eruption_pixels = np.sum(mask > 127)
            total_pixels = mask.size
            eruption_ratio = eruption_pixels / total_pixels

            if eruption_ratio > 0.35:
                risk_level = "High"
            elif eruption_ratio > 0.15:
                risk_level = "Medium"
            else:
                risk_level = "Low"
            
            results.append(f"{filename}: {risk_level} risk ({eruption_ratio*100:.2f}%)")

    with open(output_file, "w") as f:
        f.write("\n".join(results))

    print(f"[INFO] Risk analysis complete. Results saved to {output_file}")


if __name__ == "__main__":
    physics_filter_batch("predicted_masks", "risk_report.txt")
