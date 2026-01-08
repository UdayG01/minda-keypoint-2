import os
import shutil
from pathlib import Path

def merge_datasets():
    # Define source directories relative to the script location
    # Ideally absolute paths or relative to the project root
    base_dir = Path(__file__).parent.resolve()
    
    source_dirs = [
        base_dir / "new_dataset" / "frames1",
        base_dir / "new_dataset" / "frames2",
        base_dir / "new_dataset" / "frames3"
    ]
    
    # Define destination directory
    dest_dir = base_dir / "new_dataset" / "dataset"
    
    # Create destination if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Destination directory: {dest_dir}")
    
    for src_dir in source_dirs:
        if not src_dir.exists():
            print(f"Source directory {src_dir} does not exist. Skipping.")
            continue
            
        print(f"Processing source directory: {src_dir}")
        
        # Iterate over all files in the source directory
        files = list(src_dir.iterdir())
        if not files:
            print(f"  No files found in {src_dir}")
            continue

        for file_path in files:
            if file_path.is_file():
                filename = file_path.name
                dest_file_path = dest_dir / filename
                
                # Check for conflicts and rename if necessary
                if dest_file_path.exists():
                    base_name = file_path.stem
                    extension = file_path.suffix
                    counter = 1
                    
                    # Try finding a unique name
                    # We append the source folder name or just a counter to ensure uniqueness
                    # User requested: "change the name of the original file then add"
                    # We will append an underscore and counter.
                    while dest_file_path.exists():
                        new_filename = f"{base_name}_{counter}{extension}"
                        dest_file_path = dest_dir / new_filename
                        counter += 1
                        
                    print(f"  Conflict: '{filename}' exists. Renaming to '{dest_file_path.name}'")
                
                # Move the file
                try:
                    shutil.move(str(file_path), str(dest_file_path))
                    print(f"  Moved: {filename} -> {dest_file_path.name}")
                except Exception as e:
                    print(f"  Error moving {filename}: {e}")

    print("Merge completed.")

if __name__ == "__main__":
    merge_datasets()
