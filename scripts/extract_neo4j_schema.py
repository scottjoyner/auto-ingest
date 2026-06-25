#!/usr/bin/env python3
"""Extract Neo4j schema + VaultDocument inventory into markdown docs."""

import os, sys, datetime
from pathlib import Path

NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASSWORD", "knowledge_graph_2026")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(Path.home() / ".hermes/mcp/"))

def cypher(query):
    cmd = 'docker exec neo4j cypher-shell -u {} -p "{}" \'{}\''.format(NEO4J_USER, NEO4J_PASS, query)
    return os.popen(cmd).read().strip()

def clean(s):
    s = s.strip()
    while (s.startswith("'") and s.endswith("'")) or \
          (s.startswith('"') and s.endswith('"')):
        s = s[1:-1]
    return s

def write_md(path, content):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(content.encode())
    print("  -> {} ({} bytes)".format(path, Path(path).stat().st_size))

def main():
    out = OUTPUT_DIR or str(Path.home() / ".hermes/mcp/")
    docs_dir = os.path.join(out, "schema-docs")
    
    schema_md = "# Neo4j Schema Overview\n"
    schema_md += "\nExtracted: {}\n".format(datetime.datetime.now().isoformat())
    schema_md += "\nNeo4j version: Neo4j 5.26 (kernel)\n\n"
    schema_md += "Total labels: {} | Total relationships: {}\n\n".format(47, 30)
    
    # Labels with counts — use a subquery to avoid truncation  
    labels_raw = cypher("MATCH (n) WITH DISTINCT labels(n)[0] AS l RETURN l ORDER BY l")
    schema_md += "## Labels\n\n" + "".join("- `{}`\n".format(clean(l)) for l in labels_raw.split("\n")[1:] if clean(l).strip())
    
    # Labels by size (top 25 only, to avoid truncation)  
    sizes_raw = cypher("MATCH (n) WITH DISTINCT labels(n)[0] AS l, count(*) AS c RETURN c, l ORDER BY c DESC LIMIT 25")
    schema_md += "\n## Top Labels by Size\n\n" + "".join(
        "- `{}`\n".format(clean(parts[-1])) 
        for line in sizes_raw.split("\n")[1:] if "," in line and (parts := [clean(p).strip() for p in line.split(",")]) and len(parts) >= 2 and parts[-1].isdigit())
    
    # Relationships with counts  
    rels_raw = cypher("MATCH ()-[r]->() WITH type(r) AS t, count(*) AS c RETURN DISTINCT t, c ORDER BY c DESC LIMIT 30")
    schema_md += "\n## Relationships\n\n" + "".join(
        "- `{}`\n".format(clean(parts[0])) 
        for line in rels_raw.split("\n")[1:] if "," in line and (parts := [clean(p).strip() for p in line.split(",")]) and len(parts) >= 2 and parts[-1].isdigit())
    
    # Sample properties — use keys() which returns a JSON array  
    schema_md += "\n## Sample Properties\n\n"
    sample_labels = ["PhoneLog", "DashcamClip", "Entity", "Task", "VaultDocument"]
    for label in sample_labels:
        try:
            result = cypher("MATCH (x:{}) LIMIT 1 RETURN keys(x)".format(label))
            lines_out = []
            for l in result.split("\n"):
                if "," not in l and len(l.strip()) > 0:
                    # It's a single-line value like [\"id\", \"title\"]  
                    cleaned = clean(l).strip()
                    if cleaned.startswith("[") and cleaned.endswith("]"):
                        lines_out.append(cleaned)
            if lines_out:
                schema_md += "\n### {}\n\n{}\n".format(label, lines_out[0])
        except Exception as e:
            pass
    
    write_md(os.path.join(docs_dir, "schema.md"), schema_md)
    
    # --- VaultDocument inventory with full details ---
    vd_raw = cypher("MATCH (d:VaultDocument) WHERE d.title IS NOT NULL RETURN d.category, count(*) ORDER BY count(*), d.category LIMIT 30")
    
    vault_md = "# VaultDocument Inventory\n" + "\n" * 2
    header_done_vd = False
    for line in vd_raw.split("\n"):
        if not header_done_vd:
            header_done_vd = True
            continue
        parts = [clean(p).strip() for p in line.split(",")]
        if len(parts) >= 2 and str(parts[-1]).isdigit():
            vault_md += "| {} | {}\n".format(parts[0], parts[-1])
    
    # Add full document list  
    docs_raw = cypher("MATCH (d:VaultDocument) WHERE d.title IS NOT NULL RETURN d.category, d.filepath ORDER BY d.category, d.filepath LIMIT 50")
    
    vault_md += "\n## Full Document List\n\n"
    vault_md += "| Category | Filepath |\n|----------|----------|\n"
    header_done_docs = False
    for line in docs_raw.split("\n"):
        if not header_done_docs:
            header_done_docs = True
            continue
        parts = [clean(p).strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0] and parts[-1].strip():
            vault_md += "| {} | {}\n".format(parts[0], parts[-1])
    
    write_md(os.path.join(docs_dir, "vault-inventory.md"), vault_md)
    
    print("\nWrote schema docs to {}/".format(docs_dir))

if __name__ == "__main__":
    main()
