# Verdant interface prototype

Open index.html directly, or run from the project root:

    python -m http.server 8765 --directory APP/prototype

Then open http://localhost:8765.

The four views are Overview, Yield preview, Data pipeline, and Models & learning.
State-dependent district selection, form validation, navigation and preview
results work locally. District options are illustrative, not exhaustive.

The yield card uses a fixed illustrative 4.20 t/ha. Production is that value
multiplied by entered area. No yield model, weather service or DAE inference runs
inside this interface. The animation illustrates the proposed steps.
The actual fitted DAE and training CSVs live separately in the project.

Green/off-white colours, vector field art, responsive layouts and
keyboard-accessible controls. Google Fonts is optional; system fonts work offline.
