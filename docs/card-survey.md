# Real card survey (read-only, 2026-10-05)

`/media/scott/UNTITLED`, 63,813 objects after excluding `.Trashes` + `System Volume
Information`. Top-level dirs: DCIM, VIDEO, overland, PHOTO, yolo, heatmap.

| category | files | GB | % bytes | in pipeline scope? |
|---|---|---|---|---|
| media (mp4/avi/mov/mkv/m4v/mp3/wav/m4a/flac/jpg/png) | 17,020 | 91.71 | 97.3 | yes |
| csv sidecars (_YOLOv8n.csv etc) | 2,973 | 1.88 | 2.0 | yes - dashcam detection |
| json (overland/locations_*.json) | 36,076 | 0.53 | 0.6 | no - not a transcript sidecar |
| python package (.py/.pyc/.pyi/.so/.h/.f90) | 6,194 | 0.12 | 0.1 | no |
| other | 1,550 | 0.03 | 0.0 | no |

**69% of files are not pipeline input.** By bytes it is 97% media, so a
byte-blind "copy everything" is not a disaster - but it stages 43,800 files of
another program's data and location history into the archive.

## Discovery contract

`auto_ingest/ingest/transcripts.py:94-99` matches suffix patterns, case-insensitively:
`_transcription.txt`, `_transcription.csv`, `_transcription_(entites|entities).csv`,
`_speakers.rttm`, `_metadata.csv`, and media `\.(wav|mp3|m4a|flac|mp4|mov|mkv)$`.

`auto_ingest/dashcam/yolo_embeddings.py:1695-1697` is NOT case-insensitive and
hardcodes `{k}.MP4`, so a lowercase `.mp4` on a vfat card is invisible to it.

## Key derivation

`canonicalize_key` (`:245-249`) needs a 14-digit `YYYY_MMDD_HHMMSS` in the name
**or the full path** (`:236-242`, including a `YYYY/MM/DD/` directory component).
Without one it falls back to a sanitised name, then `stable_id(full_path)`.

Card reality: `DCIM/2026_0829_123850_{F,R}.MP4` is keyable. `VIDEO/MOVI0000.avi`
and every `overland/locations_*.json` are not.

## Consequences for the plan

1. Staging must **classify**, not walk-and-copy. Scope = media + csv.
2. Staging must **normalise layout** to what discovery expects, otherwise a card
   laid out `DCIM/Movie/*.MP4` is copied intact and then ignored.
3. Undated media needs an explicit decision, not a silent `stable_id`.
4. The YOLO case-sensitivity bug must be fixed or staged names are coerced.
