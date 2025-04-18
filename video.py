from pytube import YouTube
import os # Often useful for specifying download paths

# --- Get the YouTube URL ---
# Option 1: Hardcode it
video_url = "https://www.youtube.com/watch?v=1mJb-sLjIaM"

# Option 2: Get it from user input
#video_url = input("Enter the YouTube video URL: ")

# --- Define where to save the video ---
# Gets the directory where your script is running
script_dir = os.path.dirname(__file__)
# Creates a 'downloads' subfolder (optional, but good practice)
download_path = os.path.join(script_dir, "downloads")
os.makedirs(download_path, exist_ok=True) # Create the folder if it doesn't exist

try:
    print(f"Fetching video info for: {video_url}")
    yt = YouTube(video_url)

    print(f"Title: {yt.title}")
    print("Fetching available streams...")

    # --- Choose a stream ---
    # Get the highest resolution progressive stream (video+audio combined)
    # Or use .filter() for more options (e.g., file_extension='mp4', resolution='720p')
    stream = yt.streams.get_highest_resolution()
    # Example of filtering: stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()


    if stream:
        print(f"Found stream: {stream.resolution}, {stream.mime_type}")
        print(f"Downloading to: {download_path}")
        # --- Download the video ---
        stream.download(output_path=download_path)
        print("Download completed!")
    else:
        print("No suitable stream found.")

except Exception as e:
    print(f"An error occurred: {e}")