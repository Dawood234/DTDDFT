import re
import os
import argparse
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('--method',type=str,required=True)
parser.add_argument('--folder_path',type=str,required=True)
args=parser.parse_args()

def find_last_occurrence_in_file(filename, pattern):
    last_occurrence = None
    try:
        with open(filename, 'r') as file:
            for line in file:
                if re.search(pattern, line):
                    last_occurrence = line
    except FileNotFoundError:
        print(f"File {filename} not found.")
    return last_occurrence

def extract_number(line):
    match = re.search(r"E=\s*([\d.-]+)D([\d+-]+)", line)
    if match:
        number_str = f"{match.group(1)}E{match.group(2)}"
        number = float(number_str)
        return number
    else:
        return None

def process_files(folder_path, method):
    pattern = r"\s*Vector\s+18\s+Occ="
    extracted_numbers = []

    # Gather all matching files
    files = [f for f in os.listdir(folder_path) if re.match(f"{method}_input_\d+\.out", f)]
    # Sort files based on the numeric part in the filename
    files.sort(key=lambda x: int(re.search(r"(\d+)\.out$", x).group(1)))

    for filename in files:
        full_path = os.path.join(folder_path, filename)
        last_occurrence = find_last_occurrence_in_file(full_path, pattern)
        if last_occurrence:
            number = extract_number(last_occurrence)
            if number is not None:
                extracted_numbers.append(number)

    return extracted_numbers

if __name__ == "__main__":
    folder_path = args.folder_path
    method = args.method
    extracted_numbers = process_files(folder_path, method)
    
    # print("Extracted numbers in proper order:")
    # for num in extracted_numbers:
    #     print(f"{num:.8f}")
    print(f'mo18=np.array({extracted_numbers})')
    # plt.plot(extracted_numbers)
    # plt.show()
