import os
import csv
import re

def extract_roc_and_epoch_from_log(log_path):
    """
    Extract the 'test/roc' value and best epoch from the log file.
    Handles two formats for 'test/roc'.
    """
    roc_value = None
    best_epoch = None
    try:
        with open(log_path, 'r') as file:
            content = file.read()

            roc_match_primary = re.search(r"'test/roc':\s*([\d\.]+)", content)
            
            if roc_match_primary:
                roc_value = float(roc_match_primary.group(1))
            else:

                roc_match_alternative = re.search(r"Test results:\s*\[\{'test/roc':\s*([\d\.]+)\}\]", content)
                if roc_match_alternative:
                    roc_value = float(roc_match_alternative.group(1))

            # Search for the best epoch pattern (independently of ROC extraction)

            epoch_match = re.search(r"best_model_epoch_epoch=(\d+)", content)
            if epoch_match:
                best_epoch = int(epoch_match.group(1))
                
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_path}")
    except Exception as e:
        print(f"Error reading or parsing {log_path}: {e}")

    
    return roc_value, best_epoch

def process_result_test_folder(folder_path, output_csv):
    """
    Process all kernel folders, extract 'test/roc' values and best epochs, 
    and save to a CSV file.
    """
    data = []
    
    if not os.path.isdir(folder_path):
        print(f"Error: Source folder {folder_path} not found or is not a directory.")
        return

    # Iterate through all items in the folder_path
    for kernel_folder_name in os.listdir(folder_path):
        kernel_path = os.path.join(folder_path, kernel_folder_name)
        # Process only if it's a directory (assuming kernel folders are directories)
        if os.path.isdir(kernel_path):
            log_file = os.path.join(kernel_path, 'log.txt')
            if os.path.exists(log_file):
                roc_value, best_epoch = extract_roc_and_epoch_from_log(log_file)
                
                # MODIFIED Condition: Add to data if roc_value is found.
                # best_epoch can be None if not found in the log.
                if roc_value is not None:
                    data.append({'Kernel': kernel_folder_name, 'test/roc': roc_value, 'best_epoch': best_epoch})
                    if best_epoch is None:
                        print(f"Info: Found 'test/roc' ({roc_value}) in {log_file}, but 'best_epoch' was not found.")
                else:
                    # This message now means 'test/roc' was not found by either pattern.
                    print(f"Info: No 'test/roc' value found in {log_file}")
            else:
                print(f"Info: Log file 'log.txt' not found in {kernel_path}")

            
    if not data:
        print("No data extracted. CSV file will not be created or will be empty if it exists.")


    # Write data to a CSV file
    try:
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = ['Kernel', 'test/roc', 'best_epoch']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"Results saved to {output_csv}")
    except IOError as e:
        print(f"Error writing to CSV file {output_csv}: {e}")


# Example usage 
if __name__ == "__main__":
    # Path to the main folder containing kernel subfolders
    folder_path = '/sum/' 
    # Path for the output CSV file
    output_csv = '/sum/val_results.csv' 
    
    process_result_test_folder(folder_path, output_csv)
