#!/bin/bash


# Get the current directory
current_dir=$(pwd)
current_dir_data=$(pwd)/Data/ALLoc_data_share
# Recursively find all .py files in the current directory and its subdirectories
find "$current_dir" -type f -name "*.py" | while read file; do
    # Count how many occurrences of "/home/chenzirui/Desktop/home_mnt" are in the file before replacing
    replacements_in_file=$(grep -o "/home/chenzirui/Desktop/home_mnt" "$file" | wc -l)

    # If there are any replacements to make
    if [ "$replacements_in_file" -gt 0 ]; then
        # Perform the replacement in the file
        sed -i "s|/home/chenzirui/Desktop/home_mnt|$current_dir_data|g" "$file"

        # Output which file was updated and how many occurrences were replaced
        echo "Updated file: $file (replaced $replacements_in_file occurrences)"
    fi

    replacements_in_file=$(grep -o "/home/chenzirui/Desktop/ALLoc" "$file" | wc -l)
    # If there are any replacements to make
    if [ "$replacements_in_file" -gt 0 ]; then
        # Perform the replacement in the file
        sed -i "s|/home/chenzirui/Desktop/ALLoc|$current_dir|g" "$file"

        # Output which file was updated and how many occurrences were replaced
        echo "Updated file: $file (replaced $replacements_in_file occurrences)"
    fi

done

# Output the total number of replacements
echo "Replacement complete! "
