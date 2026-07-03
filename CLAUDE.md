# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SeaAlert is a **static academic research paper website** hosted on GitHub Pages. It showcases the paper "SeaAlert: Critical Information Extraction From Maritime Distress Communications with Large Language Models" (IEEE Access, 2026).

**No build process, no package manager, no backend.** Files are served directly by GitHub Pages.

## Development

To preview locally, serve the root directory with any static file server:
```bash
python3 -m http.server 8000
# or
npx serve .
```

Then open `http://localhost:8000`.

## Architecture

Single-page site (`index.html`) with custom CSS and JS:

- **[index.html](index.html)** — All page content (~1060 lines). Sections: publication header, abstract, pipeline diagram, results tables/carousels, BibTeX citation.
- **[static/css/index.css](static/css/index.css)** — All custom styling (~754 lines). Uses CSS custom properties for theming. Dark mode applied via `.dark-mode` class on `<html>`. Responsive breakpoints: 480px, 769px, 1024px.
- **[static/js/index.js](static/js/index.js)** — All interactivity (~143 lines): dark mode toggle, sidebar menu, scroll-to-top, BibTeX copy-to-clipboard, Bulma carousel/slider initialization, video autoplay via IntersectionObserver.

### Frontend Dependencies (bundled in `static/`)
- **Bulma** CSS framework + carousel + slider plugins
- **jQuery** 3.5.1
- **FontAwesome** + Academicons (icons)
- **Inter** font via Google Fonts

### Writing Style
- **Never use em dashes (`—`)** in any HTML content. Use a regular hyphen (`-`) instead.

### Key Patterns
- Dark mode: toggled by adding/removing `.dark-mode` class on `<html>`, persisted in `localStorage`
- Carousels: initialized with `bulmaCarousel.attach()` / `bulmaSlider.attach()` in `index.js`
- All external CDN assets have local fallbacks in `static/`
