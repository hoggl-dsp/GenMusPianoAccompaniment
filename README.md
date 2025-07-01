# GenMusPianoAccompaniment

A system for generating musical accompaniment to voice, whether sung, spoken or otherwise.

## Stages
```mermaid
graph TD;
    A{Audio}-->P{Pitch}
    A-->D{Descriptors}
    P-->M{Melody}
    D-->M
    M-->H{Harmonizer}
    H-->T{To MIDI}
    T-->S{Save MIDI file}
    T-->AD{Sythisize Audio}
    AD-->ME{Merge orig and synthed}
```

### Melody Extraction

### Accompaniment Generation
