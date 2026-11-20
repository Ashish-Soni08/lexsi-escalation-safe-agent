# Stage 2 paper (EMNLP/ACL template)

Build: `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main`.
Figures are PDF conversions of `docs/figures/*-light.svg` (`rsvg-convert -f pdf`).
Switch `\usepackage[final]{acl}` to `[review]` for an anonymous version.
