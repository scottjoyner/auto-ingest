#!/usr/bin/env python3
with open('/home/scott/git/auto-ingest/.env') as f: content = f.read()
old = 'NEO4J_PASSWORD=*** '
new = 'NEO4J_PASSWORD=*** 'w') as f: f.write(content)
print('Updated .env file')
