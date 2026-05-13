# No-LLM workflow example

```bash
content-os init
content-os new-run --title "Bookmarkable content systems for technical founders" --route ORIGINAL --format x_thread
content-os brief 2026-05-bookmarkable-content-systems-for-technical-founders
content-os draft 2026-05-bookmarkable-content-systems-for-technical-founders
# Edit draft-package.md to replace TODOs with human-approved copy and proof.
content-os verify 2026-05-bookmarkable-content-systems-for-technical-founders
content-os approve 2026-05-bookmarkable-content-systems-for-technical-founders
content-os scheduler-handoff 2026-05-bookmarkable-content-systems-for-technical-founders --platform x
content-os postiz-export 2026-05-bookmarkable-content-systems-for-technical-founders --out ./postiz_payload.json
```

If verification fails, improve the draft rather than forcing approval. `--force` is reserved for explicit manual exceptions recorded in `review.md`.
