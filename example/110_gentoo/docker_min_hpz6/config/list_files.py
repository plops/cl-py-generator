import glob
import os


DEFAULT_FIRMWARE_DIR = "/usr/lib/firmware"

# Exclude Nvidia GSP firmware blobs from the generated list.
EXCLUSION_PATTERNS = [
    "nvidia/*/gsp_ga10x.bin",
    "nvidia/*/gsp_tu10x.bin",
]


def list_firmware_files_glob(directory=DEFAULT_FIRMWARE_DIR):
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
    all_files = [f for f in all_files if os.path.isfile(f)]

    exclusion_patterns_glob = [os.path.join(directory, pattern) for pattern in EXCLUSION_PATTERNS]
    excluded_files = []
    for pattern in exclusion_patterns_glob:
        excluded_files.extend(glob.glob(pattern))

    filtered_files = [f for f in all_files if f not in excluded_files]
    return filtered_files


if __name__ == "__main__":
    firmware_files = list_firmware_files_glob()

    if firmware_files:
        for file_path in firmware_files:
            print(file_path[1:])
