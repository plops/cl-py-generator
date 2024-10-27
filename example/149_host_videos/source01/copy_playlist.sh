#!/bin/bash


# i have a file containing a playlist (path of one music file in each
# line, path may contain space character or paren). they are in
# different folders. i want to copy them all into a new
# folder. filename collision shall be resolved by asigning a new
# unique filename.



# Path to the playlist file
playlist_file="$1"

# Destination directory where the files will be copied
destination_dir="$2"

# Create the destination directory if it doesn't exist
mkdir -p "$destination_dir"

# Iterate over each line in the playlist file
while IFS= read -r filepath; do
    # Check if the path is not empty
    if [ -n "$filepath" ]; then
        # Extract the filename from the full path
        filename=$(basename "$filepath")

        # Create a unique destination filename to avoid collisions
        unique_filename=$(uuidgen)-$filename

        # Copy the file to the destination directory with the unique filename
        cp "$filepath" "$destination_dir/$unique_filename"
    fi
done < "$playlist_file"

echo "Files have been copied to $destination_dir"
