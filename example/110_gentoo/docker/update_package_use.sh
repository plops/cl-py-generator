#!/bin/bash

# Set the package.use file path
PACKAGE_USE_FILE="/etc/portage/package.use/package.use"

# Create a temporary file to store the updated configuration
TEMP_FILE="$(mktemp)"

# Function to get current USE flags for a package
get_current_use() {
    local package="$1"
    # equery uses "$package" | awk '{printf "%s ", $1}' | tr '[:upper:]' '[:lower:]' | sort | uniq
    equery uses "$package" | awk '/\[\+\]/{flag=$1;gsub(/[][+]/,"",flag);printf "%s ",tolower(flag)}/\[-\]/{flag=$1;gsub(/[][-]/,"",flag);printf "-%s ",tolower(flag)}' | tr -s ' ' #| sort #removed sort here, get correct -/+ state
}

# Function to normalize a line (lowercase, sort flags)
normalize_line() {
    local line="$1"
    local category_package=$(echo "$line" | cut -d ' ' -f 1)
    local flags=$(echo "$line" | cut -d ' ' -f 2-)

       if [[ -z "$flags" ]]; then
               #there are no use-flags, keep the line as is.
               echo "$line"
               return
       fi

    local sorted_flags=$(echo "$flags" | tr ' ' '\n' | sort -u | tr '\n' ' ')
    echo "$category_package $sorted_flags"
}

# Create a list of installed packages in the format category/package
# sort alphabetically, and only show the first match for each package (the highest version)
# if you have multiple slots installed, you will otherwise get multiple lines!
installed_packages=$(portageq list_installed / | cut -d '/' -f 1-2 | sort -u | awk '!seen[$0]++')


# Loop through the existing package.use file
# If entry exists in installed packages, it will be rewritten
# If not, it is kept "as is". This will keep your comments.
while IFS= read -r line; do
  # Skip empty lines and comments
  if [[ -z "$line" ]] || [[ "$line" =~ ^# ]]; then
    echo "$line" >> "$TEMP_FILE"
    continue
  fi

  package_in_file=$(echo $line | awk '{print $1}')

  # Skip if the package in the file is not installed, keeping the line
  if ! echo "$installed_packages" | grep -q -F "$package_in_file"; then
    #echo "package not installed: $package_in_file"
    echo "$line" >> "$TEMP_FILE"
    continue
  fi

  # Get the current USE flags for the package
  current_use=$(get_current_use "$package_in_file")

  # Combine to the new line
  new_line=$(normalize_line "$package_in_file $current_use")

  # Write to temp file.
  echo "$new_line" >> "$TEMP_FILE"

done < "$PACKAGE_USE_FILE"


#sort all entries by package name, but keep lines beginning with '#' or empty at their position.
#this will order packages alphabetically, but keep the file readable.
awk '
/^#/ { print; next }
/^[[:space:]]*$/ { print; next }
{
  match($0, /([^/]+)\/(.+)/, arr)
    print arr[1] "\t" arr[2] "\t" $0
  }
' "$TEMP_FILE" | sort -k1,1 -k2,2 | cut -f 3- > "${TEMP_FILE}.sorted"

# Replace the original package.use file with the updated one (after checking)
echo "Review the changes in ${TEMP_FILE}.sorted"
echo "If everything looks good, run:"
echo "  sudo mv ${TEMP_FILE}.sorted $PACKAGE_USE_FILE"
echo "  sudo rm $TEMP_FILE"
#rm "$TEMP_FILE"  # Clean up the temporary file

exit 0 #for testing

