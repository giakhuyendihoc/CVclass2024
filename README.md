# CVclass2024
## CV Final Project

### Setup
1. **Conda Environment**: Create a new environment with Python 3.9
    ```bash
    conda create -n new python=3.9
    ```
2. **Install Dependencies**: Install required packages in the Python 3.9 environment
    ```bash
    pip install -r requirements.txt
    ```

3. **Install Blender**: Install Blender and add it to your computer’s PATH (Windows).

4. **Download Dataset**: Run the `download-dataset` script to get the organ dataset (3D models listed in `as-per-organ.csv`).

    ```bash
    python download-dataset.py
    ```

### Usage
Run the following Blender command to generate images from the downloaded 3D models:

```bash
blender -b -P blender_script.py -- --object_path "E:/Khuyen/Y4 sem 1/CV/Team Project/organ_dataset/" \
    --output_dir "E:/Khuyen/Y4 sem 1/CV/Team Project/images_dataset" \
    --engine CYCLES --scale 0.8 --num_images 100 --camera_dist 1.2
