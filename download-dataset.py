import os
import time
import requests
import pandas as pd
from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options

# Set up Microsoft Edge WebDriver
edge_driver_path = "E:\\edgedriver_win64\\msedgedriver.exe"  # Replace with the correct path to msedgedriver
service = EdgeService(edge_driver_path)
options = Options()
options.add_argument("--headless")  # Run Edge in headless mode (no GUI)
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
driver = webdriver.Edge(service=service, options=options)

# Load CSV file
csv_file = "as-per-organ.csv"  # Replace with your CSV file name
data = pd.read_csv(csv_file)

# Create output directory
output_dir = "organ_dataset"
os.makedirs(output_dir, exist_ok=True)

# Function to download GLB file
def download_glb(url, output_dir, organ, sex):
    try:
        driver.get(url)
        time.sleep(1)  # Wait for the page to load

        # Find all links containing "glb" in their text
        glb_link = None
        for element in driver.find_elements(By.TAG_NAME, "a"):
            if "glb" in element.text.lower():
                glb_link = element.get_attribute('href')
                break
        
        if glb_link:
            print(f"Detected GLB download link: {glb_link}")
            
            # Download the GLB file
            glb_response = requests.get(glb_link)
            glb_path = os.path.join(output_dir, f"{organ}_{sex}", f"{organ}_{sex}.glb")
            os.makedirs(os.path.dirname(glb_path), exist_ok=True)
            with open(glb_path, 'wb') as f:
                f.write(glb_response.content)
            print(f"Downloaded GLB file: {glb_path}")
            
            # Wait for the file to be fully downloaded (in case of delayed saving)
            file_size = os.path.getsize(glb_path)
            print(f"File size: {file_size} bytes")
            time.sleep(1)  # Sleep to allow file to be saved completely
        else:
            print(f"No GLB file found at: {url}")
    except Exception as e:
        print(f"Error accessing {url}: {e}")

# Process each row in the CSV file
for _, row in data.iterrows():
    organ = row["organ"].replace(" ", "_")
    sex = row["sex"]
    url = row["url"]

    # Download GLB file
    download_glb(url, output_dir, organ, sex)

# Close the WebDriver
driver.quit()
