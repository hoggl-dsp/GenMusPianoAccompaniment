# GenMusPianoAccompaniment

A system for generating musical accompaniment to voice, whether sung, spoken or otherwise.

## Stages
```mermaid
graph TD;
    A{Audio}-->P{Pitch}<br>1. freq/frame<br>2. confidence/frame;
    A-->D{Descriptors}<br>1. loudness/frame<br>2. MFCCs? (centroid)
    D-->M{Melody}<br>
    P-->M
    
```

### Melody Extraction

### Accompaniment Generation
