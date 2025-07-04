# Song-ify

A generative music system for adding accompaniment to voice, particularly for spoken word, developed during the [Generative Music AI Workshop (2025)](https://www.upf.edu/web/mtg/generative-music-ai-workshop) in the Music Technology Group at Universitat Pompeu Fabreu in Barcelona.

The system uses a variety of techniques from digital signal processing, music theory and generative AI to extract a melody from the spoken voice, then suitably harmonise that melody in a way which adds value to the original content.

## Stages
```mermaid
flowchart TD
    A{Audio}-->P{Pitch}
    A-->D{Descriptors}
    P-->M{Melody}
    D-->M
    M-->H{Harmonizer}
    H-->T{To MIDI}
    T-->S{Save MIDI file}
    T-->AD{Sythisize Audio}
    AD-->ME{Merge orig and synthed}

    %% Highlight input section
    subgraph pitch_extraction[Pitch extraction]
        P
        D
        M
    end

    %% Highlight MIDI generation section
    subgraph harmonization[Harmonization]
        H
        T
    end

    subgraph post_processing[Postprocessing]
        S
        AD
        ME
    end

    %% Define styling for blocks
    classDef inputBlock fill:#e3f2fd,stroke:#1e88e5,stroke-width:2px,rx:8,ry:8;
    classDef midiBlock fill:#fce4ec,stroke:#d81b60,stroke-width:2px,rx:8,ry:8;
```

### Melody Extraction

### Accompaniment Generation
