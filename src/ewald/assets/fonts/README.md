# Bundled UI fonts

The GUI loads these at start (`ewald.gui.fonts.load_fonts`), so it looks the same on Windows,
macOS and Linux instead of falling back to Segoe UI, Helvetica or DejaVu. All are under the SIL
Open Font License 1.1, which allows bundling them with software; the license texts sit next to
the fonts and ship with every install.

| Files | Font | Version | Source | License |
| --- | --- | --- | --- | --- |
| `IBMPlexSans-Regular.ttf`, `-SemiBold.ttf`, `-Bold.ttf` | IBM Plex Sans (UI text) | 3.005 | github.com/IBM/plex `packages/plex-sans/fonts/complete/ttf` | `IBM-Plex-LICENSE.txt` |
| `IBMPlexMono-Regular.ttf` | IBM Plex Mono (paths in Preferences) | 2.005 | github.com/IBM/plex `packages/plex-mono/fonts/complete/ttf` | `IBM-Plex-LICENSE.txt` |
| `JetBrainsMono-Regular.ttf`, `-Bold.ttf` | JetBrains Mono (code cells, notebook tabs) | 2.305 | github.com/JetBrains/JetBrainsMono `fonts/ttf` | `JetBrainsMono-OFL.txt` |

Plots do not use these: they keep the metric-compatible Arial / Liberation Sans of
`ewald.plotstyle`, so figures match journal fonts.
