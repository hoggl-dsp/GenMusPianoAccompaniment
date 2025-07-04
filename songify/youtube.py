import yt_dlp
import os
import streamlit as st

@st.cache_resource
def download_youtube_video(video_url, output_dir):
    """
    Downloads a YouTube video using yt-dlp and saves it to the specified directory.
    
    Args:
        video_url (str): The URL of the YouTube video to download.
        output_dir (str): The directory where the downloaded video will be saved.
    """
    output_file = None
    ydl_opts = {
        'format': "best",
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'merge_output_format': 'mp4', # Ensures video and audio are merged into MP4
        'noplaylist': True, # Prevents downloading entire playlist if URL is part of one
        'verbose': False, # Set to True for more detailed output during download
    }

    try:
        print(f"Attempting to download: {video_url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            video_title = info_dict.get('title', 'Unknown Title')
            video_ext = info_dict.get('ext', 'mp4')
            output_file = os.path.join(output_dir, f"{video_title}.{video_ext}")
            
            print(f"Video '{video_title}' downloaded successfully to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

    assert output_file is not None, "Output file should not be None after download"
    assert os.path.exists(output_file), f"Downloaded file does not exist: {output_file}"
    return output_files


if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=uvsPzuriDdA"

    output_dir = os.path.join("output", "youtube")

    os.makedirs(output_dir, exist_ok=True)

    download_youtube_video(video_url, output_dir)

    

