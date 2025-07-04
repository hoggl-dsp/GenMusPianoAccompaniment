import struct
import wave
from io import BytesIO

import librosa
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import symusic
import torch
import torchaudio

from songify import utils
from songify.main import (
    HarmonyGenerationParameters,
    MelodyExtractionParameters,
    SongifyApp,
)

melody_params = MelodyExtractionParameters()
harmony_params = HarmonyGenerationParameters()

songify_app = SongifyApp()

# Page config
st.set_page_config(page_title="Audio Processing App", page_icon="🎵", layout="wide")

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
# .upload-section {
#     border: 2px dashed #ccc;
#     border-radius: 10px;
#     padding: 2rem;
#     text-align: center;
#     margin-bottom: 2rem;
#     background-color: #f9f9f9;
# }
# .control-section {
#     background-color: #f8f9fa;
#     padding: 1.5rem;
#     border-radius: 10px;
#     margin-bottom: 1rem;
# }
# .harmonizer-section {
#     background-color: #f8f9fa;
#     padding: 1.5rem;
#     border-radius: 10px;
#     margin-bottom: 1rem;
# }
.generate-button {
    display: flex;
    justify-content: center;
    margin: 2rem 0;
}
# .player-section {
#     border: 2px solid #ccc;
#     border-radius: 10px;
#     padding: 2rem;
#     text-align: center;
#     margin-bottom: 1rem;
# }
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

# Header
st.markdown('<h1 class="main-title">Songify</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Any sound can be a song XP</p>', unsafe_allow_html=True
)

# Audio Upload Section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("### 🎵 Drag&Drop")
uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3", "flac", "ogg"],
    help="Upload your audio file for processing",
    key="main_audio_uploader"
)

if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file
    # Display audio player for uploaded file
    st.audio(uploaded_file, format="audio/wav")

    try:
        # Load audio file into SongifyApp
        songify_app.load_audio(uploaded_file)

        # Waveform visualization placeholder
        st.markdown("**Waveform + Annotations**")

        # Create a simple waveform visualization
        fig, ax = plt.subplots(figsize=(12, 3))
        librosa.display.waveshow(
            songify_app.audio.numpy(), sr=songify_app.sample_rate, ax=ax, alpha=0.5
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Audio Waveform")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.error(f"Error loading audio file: {e}")
        st.session_state.uploaded_file = None
        uploaded_file = None


st.markdown("</div>", unsafe_allow_html=True)

# Main controls section
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="control-section">', unsafe_allow_html=True)
    st.markdown("### Melody Extraction")

    # Onset algorithm dropdown
    onset_algorithm = st.selectbox(
        "Onset algorithm",
        [
            "Librosa",
        ],
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

    # Frame Size slider
    frame_size = st.slider(
        "Frame Size (millis)", min_value=10, max_value=100, value=20, step=1
    )

    # Median Filter slider
    median_filter = st.slider(
        "Median Filter", min_value=1, max_value=11, value=3, step=2
    )

    # Note Duration range slider
    note_duration = st.slider(
        "Note Duration", min_value=0.0, max_value=1.0, value=(0.2, 0.8), step=0.01
    )
    st.caption(f"min: {note_duration[0]:.2f} - max: {note_duration[1]:.2f}")

    # Dry/Wet slider
    dry_wet = st.slider("Dry/Wet", min_value=0.0, max_value=1.0, value=0.6, step=0.01)

    melody_params.onset_detection = onset_algorithm
    melody_params.pitch_algorithm = pitch_algorithm
    melody_params.frame_size = frame_size
    melody_params.median_filter = median_filter
    melody_params.min_note_duration = note_duration[0]
    melody_params.max_note_duration = note_duration[1]

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
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
        "Duration Threshold (seconds)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.01,
    )

    harmony_params.chord_melody_congruence = congruence
    harmony_params.chord_variety = variety
    harmony_params.harmonic_flow = flow
    harmony_params.dissonance = dissonance
    harmony_params.cadence = cadence
    harmony_params.duration_threshold = duration_threshold
    st.markdown("</div>", unsafe_allow_html=True)

# Generate Button
st.markdown('<div class="generate-button">', unsafe_allow_html=True)
if st.button("🎵 Generate!", type="primary"):
    if st.session_state.uploaded_file is not None:
        with st.spinner("Processing audio... This may take a moment."):

            melody_score, harmony_score, melody_audio, harmony_audio = (
                songify_app.generate(
                    melody_params=melody_params, harmony_params=harmony_params
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

            # Combine and normalize
            generated_audio = utils.mix_audio(
                melody_audio, harmony_audio, blend=0.5, stereo=True
            )

            print("Generated audio shape before mixing:", generated_audio.shape)

            output_audio = utils.mix_audio(
                original_audio,
                generated_audio,
                blend=dry_wet,
                stereo=True,
            )
            st.success(f"Generated audio shape: {output_audio.shape}")

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
        # Display waveform
        fig, ax = plt.subplots(figsize=(12, 4))
        librosa.display.waveshow(
            st.session_state.generated_audio,
            sr=songify_app.sample_rate,
            ax=ax,
            alpha=0.5,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Generated Audio Waveform")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

        # Convert to audio format for playback
        audio_data = st.session_state.generated_audio
        sample_rate = songify_app.sample_rate

        st.audio(audio_data, format="audio/wav", sample_rate=sample_rate)

else:
    st.info("Generate audio to see the waveform and player")

st.markdown("</div>", unsafe_allow_html=True)

# Download Section
st.markdown('<div class="download-section">', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
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

with col2:
    if st.session_state.generated_audio is not None:
        # Create WAV file for download
        audio_data = st.session_state.generated_audio
        sample_rate = songify_app.sample_rate

        wav_buffer = BytesIO()
        torchaudio.save(wav_buffer, torch.from_numpy(audio_data), sample_rate, format="wav", bits_per_sample=16)

        st.download_button(
            label="📥 Download Wav",
            data=wav_buffer.getvalue(),
            file_name="generated_audio.wav",
            mime="audio/wav",
        )
    else:
        st.button("📥 Download Wav", disabled=True)

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("*Audio Processing App - Built with Streamlit*")
