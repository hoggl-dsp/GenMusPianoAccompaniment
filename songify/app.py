import struct
import wave
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from main import HarmonyGenerationParameters, MelodyExtractionParameters, SongifyApp

melody_params = MelodyExtractionParameters()
harmony_params = HarmonyGenerationParameters()

songify_app = SongifyApp(
    melody_params=melody_params,
    harmony_params=harmony_params,
)

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
.player-section {
    border: 2px solid #ccc;
    border-radius: 10px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1rem;
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

# Header
st.markdown('<h1 class="main-title">Title APP</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">nice catch phrase</p>', unsafe_allow_html=True)

# Audio Upload Section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("### 🎵 Drag&Drop")
uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3", "flac", "ogg"],
    help="Upload your audio file for processing",
)

if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file
    # Display audio player for uploaded file
    st.audio(uploaded_file, format="audio/wav")

    # Waveform visualization placeholder
    st.markdown("**Waveform + Annotations**")
    # Create a simple waveform visualization
    fig, ax = plt.subplots(figsize=(12, 3))
    t = np.linspace(0, 5, 1000)
    waveform = np.sin(2 * np.pi * 440 * t) * np.exp(-t / 2)
    ax.plot(t, waveform, "b-", linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Audio Waveform")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()
    songify_app.load_audio(uploaded_file)
    extracted_melody_data = songify_app.extract_melody()

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
            "Complex Domain",
            "High Frequency Content",
            "Spectral Difference",
            "Phase Deviation",
        ],
        index=0,
    )

    # Frame Size slider
    frame_size = st.slider(
        "Frame Size", min_value=0.0, max_value=1.0, value=0.5, step=0.01
    )

    # Median Filter slider
    median_filter = st.slider(
        "Median Filter", min_value=0.0, max_value=1.0, value=0.3, step=0.01
    )

    # Note Duration range slider
    note_duration = st.slider(
        "Note Duration", min_value=0.0, max_value=1.0, value=(0.2, 0.8), step=0.01
    )
    st.caption(f"min: {note_duration[0]:.2f} - max: {note_duration[1]:.2f}")

    # Dry/Wet slider
    dry_wet = st.slider("Dry/Wet", min_value=0.0, max_value=1.0, value=0.6, step=0.01)

    melody_params.onset_detection = onset_algorithm
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

    harmony_params.chord_melody_congruence = congruence
    harmony_params.chord_variety = variety
    harmony_params.harmonic_flow = flow
    harmony_params.functional_harmony = 0.5  # Static value for simplicity
    st.markdown("</div>", unsafe_allow_html=True)

# Generate Button
st.markdown('<div class="generate-button">', unsafe_allow_html=True)
if st.button("🎵 Generate!", type="primary"):
    if st.session_state.uploaded_file is not None:
        with st.spinner("Processing audio... This may take a moment."):

            data = songify_app.generate(
                melody_params=melody_params, harmony_params=harmony_params
            )
            # Simulate processing time
            import time

            time.sleep(2)

            # Generate dummy audio data for demonstration
            sample_rate = 44100
            duration = 3.0  # 3 seconds
            t = np.linspace(0, duration, int(sample_rate * duration))

            # Create a simple melody based on parameters
            frequency = 440 * (1 + congruence)  # Base frequency modified by congruence
            melody = np.sin(2 * np.pi * frequency * t) * np.exp(-t / duration * variety)

            # Add some harmonics based on flow parameter
            harmony = np.sin(2 * np.pi * frequency * 1.5 * t) * flow * 0.3

            # Combine and normalize
            generated_audio = melody + harmony
            generated_audio = generated_audio * dry_wet + np.random.normal(
                0, 0.1, len(generated_audio)
            ) * (1 - dry_wet)
            generated_audio = generated_audio / np.max(np.abs(generated_audio)) * 0.8

            st.session_state.generated_audio = generated_audio

        st.success("Audio generated successfully!")
        st.balloons()
    else:
        st.warning("Please upload an audio file first!")

st.markdown("</div>", unsafe_allow_html=True)

# Generated Audio Player Section
st.markdown('<div class="player-section">', unsafe_allow_html=True)
st.markdown("### ▶️ Generated Audio Waveform Player")

if st.session_state.generated_audio is not None:
    # Display waveform
    fig, ax = plt.subplots(figsize=(12, 4))
    t = np.linspace(0, 3, len(st.session_state.generated_audio))
    ax.plot(t, st.session_state.generated_audio, "r-", linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Generated Audio Waveform")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

    # Convert to audio format for playback
    audio_data = st.session_state.generated_audio
    sample_rate = 44100

    # Convert to 16-bit PCM
    audio_int16 = (audio_data * 32767).astype(np.int16)

    # Create WAV file in memory
    wav_buffer = BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    wav_buffer.seek(0)
    st.audio(wav_buffer.read(), format="audio/wav", sample_rate=sample_rate)

else:
    st.info("Generate audio to see the waveform and player")

st.markdown("</div>", unsafe_allow_html=True)

# Download Section
st.markdown('<div class="download-section">', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    if st.session_state.generated_audio is not None:
        # Create MIDI file (simplified - just a placeholder)
        midi_data = b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00\x60MTrk\x00\x00\x00\x0b\x00\xff/\x00"
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
        sample_rate = 44100
        audio_int16 = (audio_data * 32767).astype(np.int16)

        wav_buffer = BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

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
