# Speaker "Me" Anchor — Design & Algorithm

Problem: per-clip `SPEAKER_00`/`SPEAKER_01` are linked to `GlobalSpeaker`s, but **none are labeled as Scott**, so you are not identified in most clips. 24,077/24,506 `GlobalSpeaker`s are `tentative`.

## 1. Add identity properties to `GlobalSpeaker`
```
ALTER (once): add properties person_id (STRING), is_me (BOOLEAN), label (STRING), confirmed_by (STRING)
```
- `is_me=true` marks the dominant global speaker that is you.
- `person_id` lets us label other recurring people (contacts) later.

## 2. Bootstrap the anchor (one-time)
Input: a small curated corpus of clips where you are the **only** speaker.
- Default location: `<fileserver_root>/me/` (recordings of you reading a known script).
- Compute ECAPA centroid per clip → average → `scott_centroid` (192-d, unit).
```
scott_centroid = unit( mean( embed(clip) for clip in me_corpus ) )
```
- Find best `GlobalSpeaker` match:
```
MATCH (g:GlobalSpeaker) WHERE g.embedding IS NOT NULL
RETURN g.id, g.score = cosine(g.embedding, $scott_centroid)
ORDER BY g.score DESC LIMIT 5
```
- If top score ≥ `ANCHOR_THRESH` (e.g. 0.70): `SET g.is_me=true, g.person_id='scott', g.label='Scott'`.
- Else: create a **new** `GlobalSpeaker` from `scott_centroid` with `is_me=true` and re-link your `VoiceIdentity{user_id:'scott'}` to it (`IS_GLOBAL_SPEAKER`).

Fallback if no curated corpus: use the existing `VoiceIdentity{user_id:'scott'}` linked `Speaker`/`GlobalSpeaker` as the seed centroid.

## 3. Propagate to every clip (per run)
Extend `link_global_speakers.py` so that whenever a local `Speaker` is clustered into a `GlobalSpeaker` with `is_me=true` (or `person_id` set), the local `Speaker` also gets `speaker.is_me=true`. Then a clip resolves "who is SPEAKER_00" via:
```
MATCH (t:Transcription {key:$clip})
      -[:HAS_SEGMENT]->(s:Segment)-[:SPOKEN_BY]->(sp:Speaker)
      -[:SAME_PERSON]->(g:GlobalSpeaker)
RETURN sp.label, g.is_me, g.person_id, g.label
```
Now `SPEAKER_00` in any 1-min clip is attributable to you when applicable.

## 4. Scope by Trip (the graph query you described)
Today `DashcamClip` has **no** `Trip` link. Add it:
```
MATCH (c:DashcamClip) WHERE c.key STARTS WITH $tripPrefix
MATCH (tr:Trip {uniqueKey:$tripKey})
MERGE (c)-[:IN_TRIP]->(tr)
```
Then "isolate one person in a Trip":
```
MATCH (tr:Trip {tripId:$id})<-[:IN_TRIP]-(c:DashcamClip)
      <-[:OF_CLIP?]-(t:Transcription)-[:HAS_SEGMENT]->(s:Segment)-[:SPOKEN_BY]->(sp:Speaker)-[:SAME_PERSON]->(g:GlobalSpeaker {is_me:true})
RETURN c.key, count(*) AS scott_segments
```
This gives, per Trip, the clips where you are present — the foundation for the "brain" recall layer.

## 5. Promote tentative → confirmed
- A `GlobalSpeaker` becomes `confirmed` when: (a) `is_me`/`person_id` set, OR (b) weighted evidence `weight_sum` exceeds a threshold, OR (c) hold-out validation in `link_global_speakers.py` passes (already implemented, just not widely triggered). This directly reduces the 24,077 tentative count.

## 6. Protect the SSD
`link_global_speakers.py` currently writes snips to disk (`--write-snips`). Make snip materialization optional and default-off; compute embeddings in RAM and cache only in `emb_cache.sqlite` (already present). Avoids the SSD crash you hit with `v2.py`.
