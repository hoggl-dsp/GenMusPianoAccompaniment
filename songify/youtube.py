import yt_dlp
import os
import streamlit as st
import ffmpeg

@st.cache_resource
def download_youtube_video(video_url, output_dir, extract_audio = True):
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
    return output_file

def extract_audio_from_video(video_file: str, output_dir: str):
    """
    Extracts audio from a video file using ffmpeg.
    
    Args:
        video_file (str): The path to the video file.
        output_dir (str): The directory where the extracted audio will be saved.
    
    Returns:
        str: The path to the extracted audio file.
    """
    audio_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_file))[0]}.mp3")
    
    try:
        ffmpeg.input(video_file).output(audio_file, format='wav').run(overwrite_output=True)
        print(f"Audio extracted successfully to {audio_file}")
    except ffmpeg.Error as e:
        print(f"An error occurred while extracting audio: {e}")
    
    return audio_file

def replace_audio_in_video(video_file: str, audio_file: str, output_dir: str):
    """
    Replaces the audio in a video file with a new audio file using ffmpeg.
    
    Args:
        video_file (str): The path to the original video file.
        audio_file (str): The path to the new audio file.
        output_dir (str): The directory where the new video file will be saved.
    
    Returns:
        str: The path to the new video file with replaced audio.
    """
    new_video_file = os.path.join(output_dir, f"replaced_audio_{os.path.basename(video_file)}")
    
    try:
        video_stream = ffmpeg.input(video_file)
        audio_stream = ffmpeg.input(audio_file)
        
        ffmpeg.output(
            video_stream['v'], audio_stream['a'], 
            new_video_file, 
            vcodec='copy', 
            acodec='aac'
        ).run(overwrite_output=True)
        
        print(f"Audio replaced successfully in {new_video_file}")
    except ffmpeg.Error as e:
        print(f"An error occurred while replacing audio: {e}")
    
    return new_video_file

if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=uvsPzuriDdA"

    output_dir = os.path.join("output", "youtube")

    os.makedirs(output_dir, exist_ok=True)

    video_file = download_youtube_video(video_url, output_dir)
    audio_file = extract_audio_from_video(video_file, output_dir)
    replaced_video_file = replace_audio_in_video(video_file, audio_file, output_dir)

    print(f"Video file saved at: {video_file}")
    print(f"Audio file saved at: {audio_file}")
    print(f"Replaced video file saved at: {replaced_video_file}")

    

