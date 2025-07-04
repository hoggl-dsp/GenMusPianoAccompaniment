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
from streamlit_advanced_audio import WaveSurferOptions, audix

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
    # Display advanced audio player for uploaded file
    upload_options = WaveSurferOptions(
        wave_color="#2B88D9", progress_color="#b91d47", height=80
    )

    result = audix(uploaded_file, wavesurfer_options=upload_options)

    # Track playback status
    if result:
        if result["selectedRegion"]:
            st.write(
                f"Selected: {result['selectedRegion']['start']:.2f}s - {result['selectedRegion']['end']:.2f}s"
            )

    try:
        # Load audio file into SongifyApp
        songify_app.load_audio(uploaded_file)

        # Waveform visualization placeholder
        st.markdown("**Waveform + Annotations**")

        # Create a simple waveform visualization
        fig, ax = plt.subplots(figsize=(12, 3))
        if fig is not None and ax is not None:
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
            "rms_energy",
            "rms_flux",
            "librosa",
            "silero"
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

col1, col2, col3 = st.columns([0.5, 1, 1])

with col1:
    include_melody = st.checkbox(
        "Include Melody",
        value=True,
        help="Include the extracted melody in the output audio along with the harmony.",
    )

with col2:
    humanise_amount = st.slider(
        "Humanise (ms)",
        min_value=0,
        max_value=10,
        value=5,
        step=1,
        help="Humanise the performance of the generated audio.",
    )

with col3:
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

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("*Audio Processing App - Built with Streamlit*")
