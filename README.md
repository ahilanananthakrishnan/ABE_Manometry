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

Columns: - Time_s
- Seg_1 to Seg_75

All pressures are in mmHg.
