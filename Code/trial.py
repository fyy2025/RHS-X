import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    # Keeping the epoch argument for compatibility, defaulting to 1
    parser.add_argument('--epoch', type=int, default=1)
    args = parser.parse_args()

    output_dir = "trial_output"

    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    print(f"Generating 5 test figures in {output_dir}...")

    # Generate 5 random plots
    for i in range(1, 6):
        plt.figure(figsize=(8, 5))

        # Create random data points
        x = np.random.randn(100)
        y = np.random.randn(100)
        colors = np.random.rand(100)

        plt.scatter(x, y, c=colors, alpha=0.5, cmap='viridis')
        plt.colorbar(label="Random Intensity")

        plt.title(f"Trial Plot {i} (Job Epoch: {args.epoch})")
        plt.xlabel("X-Axis Random")
        plt.ylabel("Y-Axis Random")

        file_path = os.path.join(output_dir, f"plot_number_{i}.png")
        plt.savefig(file_path)
        plt.close()
        print(f"Successfully saved: {file_path}")

    print("All 5 trial figures have been generated.")

if __name__ == "__main__":
    main()