import os
import glob


def list_firmware_files_glob(directory="/usr/lib/firmware"):
    """
    Implementation using glob, handling wildcards more directly.
    This version is generally *less* efficient and more complex for this 
    specific problem, but demonstrates another approach.  os.walk is preferred.

    Args:
        directory: The directory to search (default: /usr/lib/firmware).

    Returns:
        A list of file paths.
    """
    all_files = glob.glob(os.path.join(directory, "**/*"), recursive=True)
    all_files = [f for f in all_files if os.path.isfile(f)] #filter out directories

    # Create exclusion patterns using glob wildcards
    exclusion_patterns_glob = [
        os.path.join(directory, "amdgpu/green_sardine*.bin"),
        os.path.join(directory, "rtl_bt/rtl8852bu*.bin"),
        os.path.join(directory, "rtw89/rtw8852b_fw-1.bin"),
        os.path.join(directory, "amd-ucode/microcode_amd_fam19h.bin"),
    ]
    
    excluded_files = []
    for pattern in exclusion_patterns_glob:
        excluded_files.extend(glob.glob(pattern))
    
    filtered_files = [f for f in all_files if f not in excluded_files]
    return filtered_files




if __name__ == "__main__":
    #firmware_files = list_firmware_files()
    firmware_files = list_firmware_files_glob() # Use the glob-based version.
    
    if firmware_files:
        #print("Firmware files (excluding specified patterns):")
        for file_path in firmware_files:
            print(file_path[1:])
    #else:
    #    print("No firmware files found (after exclusions).")
