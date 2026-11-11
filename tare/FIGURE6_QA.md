# Figure 6 QA

- Static source preflight: 21 pass, 0 warn, 0 fail; READY FOR VISUAL QA.
- Render-time panel alignment: PASS, 0 fail, 0 warn, 1.5 pt tolerance.
- PDF text audit: PASS; minimum glyph size 6.5 pt; 0 runs below 5 pt.
- Rendered collision audit: PASS; 0 text-text, text-stroke or clipping
  findings, 0 warn. Eleven fully contained in-bar labels were retained and
  visually inspected.
- Final visual inspection: panel labels, axes, direct labels, in-bar labels,
  method names and the 0.95 nominal reference line are legible at final size.

Exports:

- `Fig6_TARE.svg`
- `Fig6_TARE.pdf`
- `Fig6_TARE.tiff`
- `Fig6_TARE.png`
- `Fig6_TARE.alignment.json`
- `Fig6_TARE.collision-audit.json`
