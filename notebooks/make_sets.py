import os
from math import ceil
import shutil


def distribute_files(directory, num_sets):
    # Ensure the directory exists
    if not os.path.exists(directory):
        print("The specified directory does not exist.")
        return

    # List all .cif files in the directory
    cif_files = [f for f in os.listdir(directory) if f.endswith('.cif')]
    files_per_set = ceil(len(cif_files) / num_sets)

    # Create subdirectories and distribute files
    for i in range(num_sets):
        set_dir = os.path.join(directory, f'set_{i + 1}')
        os.makedirs(set_dir, exist_ok=True)

        # Get the slice of files for this set
        start_index = i * files_per_set
        end_index = start_index + files_per_set
        set_files = cif_files[start_index:end_index]

        # Copy files to the new subdirectory instead of moving them
        for file in set_files:
            shutil.move(os.path.join(directory, file), os.path.join(set_dir, file))

    print(f"Files distributed into {num_sets} sets.")

def main():
    this_directory = os.getcwd()
    print(this_directory)
    source_directory = os.path.join(this_directory, "extracted_files") #'..', 'data', 'pdbs')
    print(source_directory)
    target_directory = os.path.join(this_directory, 'distributed_sets')
    print(target_directory)
    os.makedirs(target_directory, exist_ok=True)

    num_of_sets = 500

    for file_name in os.listdir(source_directory):
        if file_name.endswith('.cif'):
            print("copying", file_name)
            source_file_path = os.path.join(source_directory, file_name)
            target_file_path = os.path.join(target_directory, file_name)
            shutil.copy(source_file_path, target_file_path)

    distribute_files(target_directory, num_of_sets)

if __name__ == "__main__":
    main()

