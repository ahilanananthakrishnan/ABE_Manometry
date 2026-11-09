# Whole-Colon Pressure Traces Model

`pressure_traces.py` simulates whole-colon pressure traces for three
phenotypes:

-   Healthy
-   IBS-D
-   STC

The colon is modeled as 75 coupled segments including: 
- ICC-inspired phase oscillators
- Enteric reflex drive (ascending excitation, descending inhibition)
- Smooth muscle activation dynamics
- Nonlinear pressure--area (tube law) mechanics
- A pressure-gated rectosigmoid brake in the distal colon

The script outputs time-resolved pressure (mmHg) for all 75 segments.

------------------------------------------------------------------------

## How to Run

Install the Python dependencies (Python 3.10+):

    pip install -r requirements.txt

From the project directory:

    python pressure_traces.py

------------------------------------------------------------------------

## Changing the Phenotype

At the top of `pressure_traces.py`, edit:

    PHENOTYPE = "healthy"   # options: "healthy", "ibsd", "stc"

Optional settings:

    MAKE_PLOT = True        # set False to disable plotting
    OUT_TXT = None          # custom output filename
    SEED = 171              # random seed for STC phenotype

------------------------------------------------------------------------

## Output

The script generates:

    <phenotype>_pressure_traces.txt

Columns:

-   Time_s
-   Seg_1 to Seg_75

All pressures are in mmHg.

------------------------------------------------------------------------

## Animation

`visualization.py` runs the model for the `PHENOTYPE` set in
`pressure_traces.py` and renders the internal signals (ICC drive, ENS
drive, muscle activation, tube cross-section, ...) as a video. It needs
the `ffmpeg` command-line tool, e.g.

    brew install ffmpeg        # or: conda install ffmpeg

Then run:

    python visualization.py

Choose the panels and segment range with `PANELS` and `SEGMENT_VIEW` at
the top of `visualization.py`. The video is written to

    outputs/visualization_components_<phenotype>_<segment_view>.mp4

------------------------------------------------------------------------

## Revision figures

Each script below regenerates one figure from the response to the
reviewers, written to `figures/` as PDF and SVG. Shared styling (fonts,
DPI, line widths, colors) lives in `plot_style.py`.

    python fig_phase_waveforms.py   # phase and ICC waveform vs. Eq. (1) parameters
    python fig_state_variables.py   # W, E, M, p for one proximal and one distal segment
    python fig_sensitivity.py       # local sensitivity heatmap (+ outputs/sensitivity.csv)
