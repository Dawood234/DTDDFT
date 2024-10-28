import os

def remove_files_with_extensions(extensions):
    current_directory = os.getcwd()
    for root, dirs, files in os.walk(current_directory):
        for file in files:
            for ext in extensions:
                if file.endswith(ext):
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    print(f"Removed: {file_path}")

# Extensions to remove
extensions = [".c", ".b",".db",".b^-1", ".movecs", ".0", ".p", ".zmat", ".trials"]

remove_files_with_extensions(extensions)

