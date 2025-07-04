import os
import tempfile
from io import BytesIO

import librosa
import matplotlib.pyplot as plt
import streamlit as st
import torch
import torchaudio
from streamlit_advanced_audio import WaveSurferOptions, audix

from songify import utils, youtube
from songify.main import (
    HarmonyGenerationParameters,
    MelodyExtractionParameters,
    SongifyApp,
)


def get_session_temp_dir():
    """
    Creates and manages a temporary directory for the Streamlit session.
    The directory is stored in st.session_state and will be cleaned up
    when the session ends.
    """
    if "session_data_dir" not in st.session_state:
        # Create a TemporaryDirectory object.
        # It's a context manager, but we're storing the object itself.
        # Its __exit__ method will be called on garbage collection.
        st.session_state.session_data_dir = tempfile.TemporaryDirectory()
        print(f"Created new session temporary directory: {st.session_state.session_data_dir.name}")
    return st.session_state.session_data_dir.name

melody_params = MelodyExtractionParameters()
harmony_params = HarmonyGenerationParameters()

songify_app = SongifyApp()

# Page config
st.set_page_config(page_title="Songify", page_icon="🎵", layout="centered")

# Custom CSS for styling
st.markdown(
    """
<style>
.main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: bold;
    margin-bottom: 0.5rem;
}
.sub-title {
    text-align: center;
    font-size: 1.2rem;
    color: #666;
    margin-bottom: 2rem;
}
.header-section {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
    padding: 0 1rem;
}
.about-section {
    text-align: left;
    flex: 1;
}
.github-link {
    text-align: right;
    flex: 1;
}
.github-link a {
    color: #333;
    text-decoration: none;
    font-size: 1.1rem;
}
.github-link a:hover {
    color: #0366d6;
}
.generate-button {
    display: flex;
    justify-content: center;
    margin: 2rem 0;
}
.download-section {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin-top: 1rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "generated_audio" not in st.session_state:
    st.session_state.generated_audio = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "video_file" not in st.session_state:
    st.session_state.video_file = None

# Header
st.markdown('<h1 class="main-title">Songify</h1>', unsafe_allow_html=True)
# st.markdown(
#     '<p class="sub-title">Any sound can be a song XP</p>', unsafe_allow_html=True
# )

# Header section with About and GitHub link
st.markdown(
    """
<div class="header-section">
    <div class="about-section">
        <details>
            <summary><strong>About</strong></summary>
            <p>Songify transforms any audio into musical compositions by extracting melodies and generating harmonies. Upload audio files, record from your mic, or use YouTube links to create beautiful piano accompaniments.</p>
            <h3>Team:</h3>
            <ul>
                <li>TeamMember1 - <a href="link_to_profile1" target="_blank">Linkedin/Github</a></li>
        </details>
    </div>
    <div class="github-link">
        <a href="https://github.com/your-username/songify" target="_blank">
            🐙 GitHub
        </a>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Audio Input Section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("### 🎵 Audio Input")

# Create tabs for different input types
tab1, tab2, tab3 = st.tabs(["📎 Audio File", "🎤 Mic Recording", "📺 YouTube Link"])

audio_source = None
current_audio_file = None

with tab1:
    st.markdown("**Upload or drag & drop an audio file**")
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "flac", "ogg"],
        help="Upload your audio file for processing",
        key="audio_file_uploader",
    )

    if uploaded_file is not None:
        current_audio_file = uploaded_file
        audio_source = "file"

with tab2:
    st.markdown("**Record audio using your microphone**")
    audio_value = st.audio_input("Record a voice message", key="voice_recorder")

    if audio_value is not None:
        current_audio_file = audio_value
        audio_source = "mic"

with tab3:
    st.markdown("**Load audio from YouTube (Coming Soon)**")
    youtube_url = st.text_input(
        "Enter YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="YouTube audio loading feature coming soon!",
        key="youtube_input",
    )

    if youtube_url:
        with st.spinner("Downloading Youtube Video..."):
            session_temp_dir = get_session_temp_dir()

            video_file = youtube.download_youtube_video(
                youtube_url, session_temp_dir
            )

            st.session_state.video_file = video_file

        if st.session_state.video_file:
            # Load the audio file into the app
            st.video(st.session_state.video_file, width='stretch')

            audio_file = youtube.extract_audio_from_video(video_file, session_temp_dir)
            if audio_file:
                current_audio_file = audio_file
                audio_source = "youtube"
                
                st.success(f"Downloaded YouTube video: {os.path.basename(video_file).split('.')[0]}")

# Process the selected audio source
if current_audio_file is not None:
    st.session_state.uploaded_file = current_audio_file

    # Display advanced audio player for the current audio file
    if audio_source == "file":
        upload_options = WaveSurferOptions(
            wave_color="#2B88D9", progress_color="#4ecdc4", height=80
        )
    elif audio_source == "mic":  # mic recording
        upload_options = WaveSurferOptions(
            wave_color="#e6ff67", progress_color="#4ecdc4", height=80
        )
    elif audio_source == "youtube":  # youtube audio
        upload_options = WaveSurferOptions(
            wave_color="#ff4444", progress_color="#4ecdc4", height=80
        )
    else:
        raise ValueError("Unknown audio source")

    result = audix(current_audio_file, wavesurfer_options=upload_options)

    # Track playback status
    if result:
        if result["selectedRegion"]:
            st.write(
                f"Selected: {result['selectedRegion']['start']:.2f}s - {result['selectedRegion']['end']:.2f}s"
            )

    try:
        # Load audio file into SongifyApp
        songify_app.load_audio(current_audio_file)

    except Exception as e:
        st.error(f"Error loading audio file: {e}")
        st.session_state.uploaded_file = None
        current_audio_file = None


st.markdown("</div>", unsafe_allow_html=True)

# Main controls section
melody_col, harmony_col = st.columns([1, 1])

with melody_col:
    st.markdown('<div class="control-section">', unsafe_allow_html=True)
    st.markdown("### Melody Extraction")

    # Onset algorithm dropdown
    onset_algorithm = st.selectbox(
        "Onset algorithm",
        ["rms_energy", "rms_flux", "librosa", "silero"],
        index=0,
    )

    pitch_algorithm = st.selectbox(
        "Pitch algorithm",
        [
            "Pesto",
            "Librosa",
        ],
        index=0,
    )

    # Median Filter slider
    median_filter = st.slider(
        "Median Filter", min_value=1, max_value=11, value=3, step=2
    )

    # Note Duration range slider
    note_duration = st.slider(
        "Note Duration (s)", min_value=0.02, max_value=1.0, value=(0.1, 1.0), step=0.01
    )
    st.caption(f"min: {note_duration[0]:.2f} - max: {note_duration[1]:.2f}")

    # Offset Absolute Threshold slider
    offset_absolute_threshold_db = st.slider(
        "Offset Absolute Threshold (dB)",
        min_value=-60.0,
        max_value=-12.0,
        value=-40.0,
        step=0.1,
        help="Threshold for note loudness detection in dB",
    )

    offset_relative_threshold_db = st.slider(
        "Offset Relative Threshold (dB)",
        min_value=-12.0,
        max_value=-2.0,
        value=-6.0,
        step=0.1,
        help="Relative threshold for note offset detection in dB",
    )

    melody_params.onset_detection = onset_algorithm
    melody_params.pitch_algorithm = pitch_algorithm
    melody_params.median_filter = median_filter
    melody_params.min_note_duration = note_duration[0]
    melody_params.max_note_duration = note_duration[1]
    melody_params.offset_absolute_threshold_db = offset_absolute_threshold_db
    melody_params.offset_relative_threshold_db = offset_relative_threshold_db

    st.markdown("</div>", unsafe_allow_html=True)

with harmony_col:
    st.markdown('<div class="harmonizer-section">', unsafe_allow_html=True)
    st.markdown("### Harmonizer")

    # Harmonizer float inputs
    congruence = st.number_input(
        "congruence: float",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        format="%.3f",
    )

    variety = st.number_input(
        "variety: float",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.01,
        format="%.3f",
    )

    flow = st.number_input(
        "flow: float", min_value=0.0, max_value=1.0, value=0.7, step=0.01, format="%.3f"
    )
    dissonance = st.number_input(
        "dissonance: float",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.01,
        format="%.3f",
    )

    cadence = st.number_input(
        "cadence: float",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.01,
        format="%.3f",
    )

    duration_threshold = st.slider(
        "Chord Duration Threshold (seconds)",
        min_value=0.05,
        max_value=0.2,
        value=0.1,
        step=0.01,
    )

    population_size = st.slider(
        "Initial Population Size",
        min_value=10,
        max_value=100,
        value=75,
        step=1,
    )
    generations = st.slider(
        "Number of Generations ",
        min_value=100,
        max_value=1000,
        value=750,
        step=50,
    )

    harmony_params.chord_melody_congruence = congruence
    harmony_params.chord_variety = variety
    harmony_params.harmonic_flow = flow
    harmony_params.dissonance = dissonance
    harmony_params.cadence = cadence
    harmony_params.duration_threshold = duration_threshold
    harmony_params.population_size = population_size
    harmony_params.generations = generations
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

output_col1, output_col2, output_col3 = st.columns([0.7, 1, 1])

with output_col1:
    include_melody = st.checkbox(
        "Include Melody",
        value=True,
        help="Include the extracted melody in the output audio along with the harmony.",
    )

with output_col2:
    humanise_amount = st.slider(
        "Humanise (ms)",
        min_value=0,
        max_value=10,
        value=5,
        step=1,
        help="Humanise the performance of the generated audio.",
    )

with output_col3:
    dry_wet = st.slider(
        "Dry / Wet",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.01,
        help="Adjust the balance between the original audio and the generated audio.",
    )

st.markdown("---")

# Generate Button
st.markdown('<div class="generate-button">', unsafe_allow_html=True)
if st.button("🎵 Generate!", type="primary", use_container_width=True):
    if st.session_state.uploaded_file is not None:
        with st.spinner("Processing audio... This may take a moment."):

            melody_score, harmony_score, melody_audio, harmony_audio = (
                songify_app.generate(
                    melody_params=melody_params,
                    harmony_params=harmony_params,
                    humanise=humanise_amount / 1000.0,  # Convert ms to seconds
                )
            )

            # Generate dummy audio data for demonstration
            original_audio = songify_app.audio
            sample_rate = songify_app.sample_rate

            assert original_audio is not None, "Audio data is not loaded."
            assert sample_rate is not None, "Sample rate is not set."

            print("Original audio shape:", original_audio.shape)
            print("Melody audio shape:", melody_audio.shape)
            print("Harmony audio shape:", harmony_audio.shape)

            if include_melody:
                generated_audio = utils.mix_audio(
                    melody_audio, harmony_audio, blend=0.5, stereo=True
                )
            else:
                generated_audio = harmony_audio

            print("Generated audio shape before mixing:", generated_audio.shape)

            output_audio = utils.mix_audio(
                original_audio,
                generated_audio,
                blend=dry_wet,
                stereo=True,
            )

            st.session_state.melody_score = melody_score
            st.session_state.harmony_score = harmony_score

            st.session_state.generated_audio = output_audio.numpy()

        st.success("Audio generated successfully!")
        st.balloons()
    else:
        st.warning("Please upload an audio file first!")

st.markdown("</div>", unsafe_allow_html=True)

# Generated Audio Player Section
st.markdown('<div class="player-section">', unsafe_allow_html=True)
st.markdown("### ▶️ Generated Audio Waveform Player")

if st.session_state.generated_audio is not None:
    with st.spinner("Preparing audio visualisation playback. Let it cook..."):
        # Convert to audio format for playback
        audio_data = st.session_state.generated_audio
        sample_rate = songify_app.sample_rate

        # Create advanced audio player for generated audio
        generated_options = WaveSurferOptions(
            wave_color="#ff6b6b", progress_color="#4ecdc4", height=120
        )

        # Convert numpy array to BytesIO for audix
        wav_buffer = BytesIO()
        torchaudio.save(
            wav_buffer,
            torch.from_numpy(audio_data),
            sample_rate,
            format="wav",
            bits_per_sample=16,
        )
        wav_buffer.seek(0)

        result = audix(wav_buffer, wavesurfer_options=generated_options)

        # Track playback status for generated audio
        if result:
            if result["selectedRegion"]:
                st.write(
                    f"Selected: {result['selectedRegion']['start']:.2f}s - {result['selectedRegion']['end']:.2f}s"
                )

else:
    st.info("Generate audio to see the waveform and player")

st.markdown("</div>", unsafe_allow_html=True)

# Download Section
st.markdown('<div class="download-section">', unsafe_allow_html=True)
midi_download_col, audio_download_col, video_download_col = st.columns(3)

with midi_download_col:
    if st.session_state.generated_audio is not None:
        # Create MIDI file (simplified - just a placeholder)
        score = utils.merge_scores(
            [st.session_state.melody_score, st.session_state.harmony_score]
        )

        midi_data = score.dumps_midi()

        st.download_button(
            label="📥 Download Midi",
            data=midi_data,
            file_name="generated_audio.mid",
            mime="audio/midi",
        )
    else:
        st.button("📥 Download Midi", disabled=True)

with audio_download_col:
    if st.session_state.generated_audio is not None:
        # Create WAV file for download
        audio_data = st.session_state.generated_audio
        sample_rate = songify_app.sample_rate

        wav_buffer = BytesIO()
        torchaudio.save(
            wav_buffer,
            torch.from_numpy(audio_data),
            sample_rate,
            format="wav",
            bits_per_sample=16,
        )

        st.download_button(
            label="📥 Download Wav",
            data=wav_buffer.getvalue(),
            file_name="generated_audio.wav",
            mime="audio/wav",
        )
    else:
        st.button("📥 Download Wav", disabled=True)

with video_download_col:
    if st.session_state.get("video_file") is not None and st.session_state.get("generated_audio") is not None:
        video_file = st.session_state.video_file

        new_audio = st.session_state.generated_audio
        new_audio_file = os.path.join(get_session_temp_dir(), "new_audio.wav")
        torchaudio.save(
            new_audio_file,
            torch.from_numpy(new_audio),
            songify_app.sample_rate,
            format="wav",
            bits_per_sample=16,
        )

        new_video_file = youtube.replace_audio_in_video(
            video_file, new_audio_file, get_session_temp_dir()
        )

        if new_video_file:
            st.download_button(
                label="📥 Download Video",
                data=open(new_video_file, "rb").read(),
                file_name=f"replaced_audio_{os.path.basename(video_file)}",
                mime="video/mp4",
            )
        else:
            st.error("Failed to replace audio in video.")
    
    else:
        st.button("📥 Download Video", disabled=True)

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("**Developed with ❤️ by the Songify team**")
