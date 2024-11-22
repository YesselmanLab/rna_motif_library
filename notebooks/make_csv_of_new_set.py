
import os
import csv

def list_cif_files(directory):
    """ List the names of all .cif files in the directory, excluding the file extension. """
    cif_names = [os.path.splitext(file)[0] for file in os.listdir(directory) if file.endswith('.cif')]
    return cif_names

def write_to_csv(file_names, output_file):
    """ Write the list of file names to a CSV file with the specific format required. """
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        for name in file_names:
            # Format each row as requested
            formatted_row = ["blank", f"{name}|0|0", "blank"]
            writer.writerow(formatted_row)

def main():
    directory_path = os.path.join('..', 'data', 'pdbs')

    #directory_path = input("Enter the directory path: ")  # Prompt user to enter the directory path
    cif_names = list_cif_files(directory_path)
    output_file = os.path.join(directory_path, 'new_set_cif_names.csv')  # Save the CSV in the same directory
    write_to_csv(cif_names, output_file)
    print(f"Exported CIF file names to {output_file}")

if __name__ == "__main__":
    main()
