import re

log_file = "dependency_tree.log"
packages = []

# Regex to strip ANSI codes
ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

# Regex to match lines like:
# [ebuild  N     ] www-client/firefox-bin-128.0::gentoo ... 123456 KiB
# We look for "KiB" at the end of the line (or near end) and the package name.
# Typical line: [ebuild  N     ] category/pkg-ver::repo ... 123 KiB

line_regex = re.compile(r'\]\s+(\S+)\s+.*?\s+(\d+)\s+KiB')

with open(log_file, 'r') as f:
    for line in f:
        # Strip ANSI
        clean_line = ansi_escape.sub('', line).strip()
        
        match = line_regex.search(clean_line)
        if match:
            pkg_name = match.group(1)
            size_kib = int(match.group(2))
            packages.append((pkg_name, size_kib))

# Sort by size descending
packages.sort(key=lambda x: x[1], reverse=True)

print(f"Total packages found: {len(packages)}")
print("-" * 60)
print(f"{'Size (MiB)':<12} | {'Package Name'}")
print("-" * 60)
for pkg, size in packages[:20]:
    print(f"{size/1024:<12.2f} | {pkg}")
