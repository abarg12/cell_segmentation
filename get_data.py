import os
import urllib.request
import zipfile

def download_bbbc038_test_data(download_dir="data"):
    os.makedirs(download_dir, exist_ok=True)
    
    # URLs for the test data and the solution CSV
    test_zip_url = "https://data.broadinstitute.org/bbbc/BBBC038/stage1_test.zip"
    solution_csv_url = "https://data.broadinstitute.org/bbbc/BBBC038/stage1_solution.csv"
    
    test_zip_path = os.path.join(download_dir, "stage1_test.zip")
    test_extract_path = os.path.join(download_dir, "stage1_test")
    solution_csv_path = os.path.join(download_dir, "stage1_solution.csv")
    
    # Download and extract the test images
    if not os.path.exists(test_extract_path):
        print(f"Downloading test images from {test_zip_url}...\n")
        try:
            urllib.request.urlretrieve(test_zip_url, test_zip_path)
            print("Extracting test images...")
            os.makedirs(test_extract_path, exist_ok=True)
            with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
                zip_ref.extractall(test_extract_path)
            print(f"Test images extracted to: {test_extract_path}")
        except Exception as e:
            print(f"An error occurred while downloading/extracting test images: {e}")
        finally:
            # Clean up the zip file to save local storage space
            if os.path.exists(test_zip_path):
                os.remove(test_zip_path)
                print("Cleaned up temporary test zip file.")
    else:
        print(f"Test images already exist at: {test_extract_path}")

    # Download the solution CSV (Ground Truth Annotations)
    if not os.path.exists(solution_csv_path):
        print(f"\nDownloading solution CSV from {solution_csv_url}...\n")
        try:
            urllib.request.urlretrieve(solution_csv_url, solution_csv_path)
            print(f"Solution CSV saved to: {solution_csv_path}")
        except Exception as e:
            print(f"An error occurred while downloading the solution CSV: {e}")
    else:
        print(f"\nSolution CSV already exists at: {solution_csv_path}")

if __name__ == "__main__":
    download_bbbc038_test_data()
