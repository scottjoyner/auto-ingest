/media/deathstar/UNTITLED/VIDEO/20250330071826_000013.MP4.

[media_copy] ➡️  Copying (VIDEO): 20250330071826_000013.MP4 → /mnt/8TB_2025/fileserver/bodycam/2025/03/30/video
[media_copy] 🎧 Extracting MP3: 20250330071826_000013.MP4 → /mnt/8TB_2025/fileserver/audio/2025/03/30/20250330071826_000013.mp3
[mov,mp4,m4a,3gp,3g2,mj2 @ 0x58ab184caf00] moov atom not found
[in#0 @ 0x58ab184cae00] Error opening input: Invalid data found when processing input
Error opening input file /media/deathstar/UNTITLED/VIDEO/20250330071826_000013.MP4.
Error opening input files: Invalid data found when processing input
[media_copy] ❌ ffmpeg failed on: /media/deathstar/UNTITLED/VIDEO/20250330071826_000013.MP4
[media_copy] ➡️  Copying (VIDEO): 20250331000727_000015.MP4 → /mnt/8TB_2025/fileserver/bodycam/2025/03/31/video


//Example Cypher: nearest embeddings by location only
// if you create a separate vector index on e.loc_vec (7-D)
CALL db.index.vector.queryNodes('DashcamLocIndex', 50, $probeLocVec) YIELD node, score
RETURN node.id AS eid, node.key, node.view, node.lat, node.lon, node.mph, score
ORDER BY score DESC LIMIT 20;


//Example Cypher: search by scene+location together
// probeVec already concatenated (scene probe ++ location probe)
CALL db.index.vector.queryNodes('DashcamSceneIndex', 50, $probeVec) YIELD node, score
WHERE node.level = 'second'
RETURN node.key, node.view, node.t0 AS sec, node.lat, node.lon, node.mph, score
ORDER BY score DESC LIMIT 20;


python organize_by_timestamp.py -i /mnt/8TB_2025/fileserver/dashcam/audio -o /mnt/8TB_2025/fileserver/dashcam -n -r
