import requests
import argparse

# Note: I tried to create a youtube api key and download captions
# using the youtube api, but it seems that the youtube api does not
# allow access to captions of videos that are not owned by the api key owner
# https://stackoverflow.com/questions/41087864/youtube-api-v3-download-captions-from-third-party-videos-without-asking-for-au

# I tried the api explorer using my own videos with transcript and always get a 404 Not Found error
# So maybe i still misunderstand the youtube api

# here they discuss that v3 youtube api requires consent from the owner to download captions
# https://stackoverflow.com/questions/30514097/unable-to-download-closed-captions-in-youtube-api-v3

# https://pypi.org/project/youtube-transcript-api/ uses undocumented parts of the youtube api,
# so it might be against the youtube terms of service just as yt-dlp

# https://stackoverflow.com/questions/69937867/google-video-no-longer-able-to-retrieve-captions/70013529#70013529
# here is an old api that uses protobuf encoding

# this seems to be an official example from google
# https://github.com/youtube/api-samples/blob/master/python/captions.py
# annoyingly it is python 2

def get_video_transcript(api_key, video_id):
    """Downloads the transcript of a YouTube video using the YouTube Data API v3 (REST).

    Args:
        api_key: Your YouTube Data API key.
        video_id: The ID of the YouTube video.

    Returns:
        A list of dictionaries, each representing a caption part with 'text' and 'start' timestamp,
        or None if no transcript is available or an error occurs.
    """

    base_url = "https://www.googleapis.com/youtube/v3/captions"
    params = {
        'part': 'snippet',
        'videoId': video_id,
        'key': api_key
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes

        data = response.json()

        if 'items' in data:
            # Get the first English caption track if available
            for item in data['items']:
                if item['snippet']['language'] == 'en':
                    caption_id = item['id']
                    break
            else:
                return None  # No English captions found

            # Download the caption track
            caption_url = f"{base_url}/{caption_id}"
            caption_response = requests.get(caption_url, params={'key': api_key})
            caption_response.raise_for_status()

            caption_data = caption_response.json()
            if 'items' in caption_data:
                return [
                    {'text': item['snippet']['text'], 'start': float(item['snippet']['startMs']) / 1000}
                    for item in caption_data['items']
                ]
            else:
                return None  # Caption track has no items
        else:
            return None  # No captions available for this video

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# read api key from /home/martin/yt_api_key.txt
with open("/home/martin/yt_api_key.txt") as f:
    api_key = f.read().strip()

# parse command line argument
# single positional argument like "DmgqT9CQSf8" required to download the youtube video transcript with that id
argparser = argparse.ArgumentParser()
argparser.add_argument("youtube", help="download youtube video transcript with that id")
args = argparser.parse_args()

# Example usage
video_id = args.youtube

transcript = get_video_transcript(api_key, video_id)

if transcript:
    for part in transcript:
        print(f"{part['start']:.2f}s: {part['text']}")
else:
    print("No transcript available for this video.")