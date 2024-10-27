#!/usr/bin/env python

# i have two files. one contains a list of pathnames to sound files (playlist).
# the other file contains in each line a partial filename that can match one or
# more of the pathnames. create a python script that creates a new playlist
# containing only pathnames that contain the partial names listed in the second
# file.

import argparse

def create_filtered_playlist(playlist_file, partial_names_file, output_file):
    # Read the playlist file and store pathnames in a list
    with open(playlist_file, 'r') as f:
        full_pathnames = [line.strip() for line in f]

    # Read the partial names file and store them in a set (for faster lookup)
    with open(partial_names_file, 'r') as f:
        partial_names = {line.strip().lower() for line in f}

    # Filter the pathnames that contain any of the partial names
    filtered_pathnames = [
        pathname for pathname in full_pathnames
        if any(partial_name.lower() in pathname.lower() for partial_name in partial_names)
    ]

    # Write the filtered playlist to a new file
    with open(output_file, 'w') as f:
        for pathname in filtered_pathnames:
            f.write(pathname + '\n')


# # Example usage
# playlist_file = 'playlist.txt'
# partial_names_file = 'partial_names.txt'
# output_file = 'filtered_playlist.txt'
#
# create_filtered_playlist(playlist_file, partial_names_file, output_file)

parser = argparse.ArgumentParser(description="Filter sound files based on partial names.")
parser.add_argument('playlist_file', type=str, help='Path to the playlist file containing full pathnames')
parser.add_argument('partial_names_file', type=str, help='Path to the file containing partial filenames to match')
parser.add_argument('output_file', type=str, help='Path to the output file for the filtered playlist')

args = parser.parse_args()

create_filtered_playlist(args.playlist_file, args.partial_names_file, args.output_file)
