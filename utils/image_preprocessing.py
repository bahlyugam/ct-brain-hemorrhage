import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from glob import glob

class SimpleDicomViewer:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.dicom_files = []
        self.slices = []
        self.current_index = 0
        self.hu_data = []
        self.window_center = 40
        self.window_width = 400
    
    def load_dicom_series(self):
        """Load all DICOM files from the specified folder."""
        print(f"Loading DICOM files from {self.folder_path}...")

        dicom_files = glob(os.path.join(self.folder_path, "*.dcm"))
        if not dicom_files:
            dicom_files = glob(os.path.join(self.folder_path, "*.DCM"))

        if not dicom_files:
            print(f"No DICOM files found in {self.folder_path}")
            return False

        print(f"Found {len(dicom_files)} DICOM files")

        slices = []
        for file_path in dicom_files:
            try:
                dicom_data = pydicom.dcmread(file_path, force=True)  # Force read even if compressed
                slices.append(dicom_data)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        # Sort slices by position or instance number
        try:
            slices.sort(key=lambda x: x.ImagePositionPatient[2])
        except (AttributeError, KeyError):
            try:
                slices.sort(key=lambda x: x.InstanceNumber)
            except (AttributeError, KeyError):
                print("Warning: Could not sort slices by position or instance number")

        self.slices = slices
        self.dicom_files = dicom_files

        # Convert to Hounsfield Units (HU)
        self.hu_data = []
        for s in slices:
            print(f"Processing file: {s.filename}")
            print(f"Transfer Syntax UID: {s.file_meta.TransferSyntaxUID}")

            # Check if the image is compressed
            if s.file_meta.TransferSyntaxUID.is_compressed:
                print("⚠️ Image is compressed! Using `pylibjpeg` for decompression...")
                try:
                    s.decompress()  # This will now work with `pylibjpeg`
                except Exception as e:
                    print(f"❌ Decompression failed: {e}")
                    continue  # Skip this file if decompression fails

            # Now it's safe to access pixel_array
            try:
                pixel_dtype = s.pixel_array.dtype  # Get the original data type
            except Exception as e:
                print(f"❌ Error accessing pixel array: {e}")
                continue  # Skip this file if error persists

            # Ensure correct dtype conversion
            if pixel_dtype == np.uint16:
                pixel_array = s.pixel_array.astype(np.int32)  # Convert safely to avoid overflow
                pixel_array[pixel_array > 32767] -= 65536  # Convert to int16 range
            else:
                pixel_array = s.pixel_array.astype(np.float32)

            # Apply HU scaling if present
            if hasattr(s, 'RescaleSlope') and hasattr(s, 'RescaleIntercept'):
                pixel_array = pixel_array * float(s.RescaleSlope) + float(s.RescaleIntercept)

            self.hu_data.append(pixel_array)

        return True

    def apply_window(self, image):
        """Apply windowing to the image."""
        min_value = self.window_center - self.window_width // 2
        max_value = self.window_center + self.window_width // 2
        windowed = np.clip(image, min_value, max_value)
        windowed = (windowed - min_value) / (max_value - min_value)
        return windowed
    
    def print_metadata(self, index):
        """Print the metadata for the current slice."""
        if 0 <= index < len(self.slices):
            s = self.slices[index]
            print("\nDICOM Metadata:")
            print(f"File: {self.dicom_files[index]}")
            
            # Try to print common metadata
            metadata_fields = [
                'PatientID', 'PatientName', 'PatientBirthDate', 'PatientSex',
                'StudyDescription', 'StudyDate', 'StudyTime',
                'SeriesDescription', 'SeriesNumber', 'Modality',
                'SliceThickness', 'SliceLocation', 'InstanceNumber',
                'ImagePositionPatient', 'ImageOrientationPatient',
                'PixelSpacing', 'Rows', 'Columns',
                'RescaleSlope', 'RescaleIntercept'
            ]
            
            for field in metadata_fields:
                if hasattr(s, field):
                    print(f"{field}: {getattr(s, field)}")
    
    def display_image(self, index):
        """Display the image at the given index."""
        if 0 <= index < len(self.slices):
            # Get current image data
            image_data = self.hu_data[index]
            
            # Apply windowing
            windowed_data = self.apply_window(image_data)
            
            # Create a new figure
            plt.figure(figsize=(10, 8))
            plt.imshow(windowed_data, cmap='gray')
            plt.title(f"Slice {index+1}/{len(self.slices)}")
            plt.axis('off')
            
            # Add file info as text
            plt.figtext(0.5, 0.01, f"File: {os.path.basename(self.dicom_files[index])}", 
                        ha='center', fontsize=10)
            
            # Add window info
            plt.figtext(0.01, 0.01, f"Window: C:{self.window_center} W:{self.window_width}", 
                        fontsize=10)
            
            # Show the image
            plt.tight_layout()
            plt.show(block=False)
            
            # Print metadata
            self.print_metadata(index)
            
            return True
        else:
            print("Invalid index")
            return False
    
    def start_viewer(self):
        """Start the simple viewer."""
        if not self.slices:
            print("No DICOM data loaded.")
            return
        
        index = 0
        while True:
            # Display current image
            self.display_image(index)
            
            # Get user input for navigation
            print("\nNavigation:")
            print("  n: Next slice")
            print("  p: Previous slice")
            print("  j: Jump to slice")
            print("  w: Change window settings")
            print("  q: Quit")
            
            cmd = input("Command: ").strip().lower()
            
            if cmd == 'n':
                index = min(index + 1, len(self.slices) - 1)
            elif cmd == 'p':
                index = max(index - 1, 0)
            elif cmd == 'j':
                try:
                    new_index = int(input("Enter slice number (1 to {}): ".format(len(self.slices))))
                    index = max(0, min(new_index - 1, len(self.slices) - 1))
                except ValueError:
                    print("Invalid input. Please enter a number.")
            elif cmd == 'w':
                try:
                    self.window_center = int(input("Enter window center: "))
                    self.window_width = int(input("Enter window width: "))
                except ValueError:
                    print("Invalid input. Using default window settings.")
                    self.window_center = 40
                    self.window_width = 400
            elif cmd == 'q':
                plt.close('all')
                break
            else:
                print("Invalid command")
            
            # Close previous figure
            plt.close()

# Example usage
if __name__ == "__main__":
    folder_path = "244349529 AnonymousPatient/SCMC2023051387 CT Scan Brain Plain/CT Thin Plain"
    
    viewer = SimpleDicomViewer(folder_path)
    if viewer.load_dicom_series():
        viewer.start_viewer()