#!/usr/bin/env python3
"""Fix typo in filenames: 'poision' -> 'poison' under the results directory."""
 
import os
import sys
 
 
def rename_files(results_dir: str = "results") -> None:
    if not os.path.isdir(results_dir):
        print(f"Error: Directory '{results_dir}' not found.")
        print(f"Usage: python {sys.argv[0]} [path/to/results]")
        sys.exit(1)
 
    count = 0
 
    for dirpath, dirnames, filenames in os.walk(results_dir):
        for filename in filenames:
            if "poision" in filename:
                new_filename = filename.replace("poision", "poison")
                old_path = os.path.join(dirpath, filename)
                new_path = os.path.join(dirpath, new_filename)
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path}  ->  {new_path}")
                count += 1
 
    print(f"\nDone. {count} file(s) renamed.")
 
 
if __name__ == "__main__":
    
    rename_files("data/results")