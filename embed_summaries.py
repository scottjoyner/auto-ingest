#!/usr/bin/env python3
"""Embed summaries in chunks using Neo4j APOC batch writes."""
import sys, os
sys.path.insert(0, '/home/scott/git/auto-ingest')
os.environ['NEO4J_PASSWORD'] = 'knowledge_graph_2026'

from neo4j import GraphDatabase
d = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j','knowledge_graph_2026'))

# Phase 1: Get all summaries needing embeddings, batch them  
with d.session() as s1:
    r = s1.run('''MATCH (t:Transcription)-[:HAS_SUMMARY]->(s) 
                 WHERE s.embedding IS NULL AND size(s.text)>10 
                 RETURN s.id AS summary_id, t.key AS trans_key, s.text AS text''')

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    batch_size = 50
    ids_to_embed = []
    
    for rec in r:
        txt = str(rec['text'] or '').strip()
        if len(txt) >= 10:
            ids_to_embed.append({
                'id': rec['summary_id'],
                'tkey': rec.get('trans_key') or '',
                'text': txt
            })

    total = len(ids_to_embed)
    print("Processing {} summaries...".format(total))

    # Process in chunks of batch_size  
    processed = 0
    for chunk_start in range(0, total, batch_size):
        chunk_end = min(chunk_start + batch_size, total)
        chunk_items = ids_to_embed[chunk_start:chunk_end]
        
        texts = [item['text'] for item in chunk_items]
        
        # Embed this chunk  
        import numpy as np
        embeddings = model.encode(texts)  # list of arrays
        
        with d.session() as s2:
            # Use APOC or individual MATCH+SET  
            for j, (item, emb) in enumerate(zip(chunk_items, embeddings)):
                s2.run('''MATCH (s:Summary {id:$sid}) SET s.embedding = $emb''',
                       sid=item['id'], emb=list(emb))

        processed += chunk_end - chunk_start
        if processed % 500 == 0 or processed == total:
            print("  Processed {} / {}".format(processed, total))

d.close()
print("Done!")
