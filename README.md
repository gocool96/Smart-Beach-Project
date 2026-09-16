# Smart Beach: Weather Forecasting for a Municipal Beach-Safety Pilot

Applied machine learning work on a Municipal Innovation Council (MIC) pilot in
Bruce County, Ontario. Phase 6, Winter 2025, as part of a student team at
Georgian College.

**About the code here.** The two Phase 6 notebooks are in `notebooks/`:

- `Smart_Beach-Phase_6.ipynb` -- the team's shared Colab notebook, covering the
  data preparation and the multivariate LSTM.
- `Snowfall_XGBoost_Model.ipynb` -- my gradient-boosting baseline and its
  hyperparameter tuning.

They are published as they stood at the end of the phase, not cleaned up
afterwards, so the limitations described further down are visible in them --
including the scaling issue and the cells that were run out of order. The
underlying dataset is not included, and the wider programme's confidential
material is not part of this work.

Full write-up: `Smart Beach Final Report.pdf` in this repository.

---

## The problem

Smart Beach is a three-year MIC initiative to improve beach safety in Bruce
County using applied research and technology. The wider programme covers
computer vision for crowd density and drowning detection; **our phase laid the
groundwork for the weather-prediction component** at Port Elgin and Southampton
beaches.

The question for this phase: can we forecast next-day beach conditions well
enough to be useful as an early warning, using publicly available weather
observations?

---

## Data

Daily weather observations from Wiarton Airport, the closest long-record station
to both beaches.

- **2,339 days**, 2019 to 2025
- Average, minimum and maximum temperature, precipitation, wind speed, wind
  direction, atmospheric pressure

Cleaning: columns that were almost entirely empty were dropped. Remaining gaps
were filled by linear interpolation with a column-mean fallback, so no days were
lost from the series.

---

## Feature engineering

**Wind direction as sine and cosine.** Wind direction is circular: 359 degrees
and 1 degree are nearly the same direction, but as raw numbers they sit at
opposite ends of the range. Encoding direction as its sine and cosine keeps that
adjacency intact, so the model is not told that a small shift in wind is a large
change.

**Binary precipitation flag** alongside the measured amount, separating "did it
rain at all" from "how much".

---

## Modelling

### Multivariate LSTM (team-built)

- Sliding windows: the previous **7 days** predict the next day
- 8 input features, min-max scaled
- LSTM(64) -> dropout 0.2 -> dense(32) -> 8 outputs
- Adam optimiser, MSE loss, early stopping on validation loss
- **Chronological** 70 / 15 / 15 train / validation / test split

The split is chronological on purpose. A random split on a time series lets the
model train on days that fall after the days it is tested on, which inflates
results.

Test error came out at MAE 0.1618 -- but on the scaled 0-1 values rather than
degrees or hPa, which is one of the limitations recorded below.

### XGBoost baseline (my contribution)

A gradient-boosted regression baseline predicting the snow measurement from the
other same-day weather readings, tuned with randomised search over 30 parameter
combinations with 3-fold cross-validation.

**Result: R-squared of about 0.06.** The model explained roughly 6% of the
variation, which is a genuinely useful negative result: same-day weather
readings alone are close to worthless for this target. Any real attempt needs
preceding days, seasonality, and probably a different target definition -- the
field used measures snow on the ground rather than new snowfall, and depth
carries over from previous days.

---

## What we found

The honest summary is that the model tracked temperature reasonably and wind
poorly, and working out why was the most valuable part of the phase:

1. **One model was asked to predict eight different things.** Temperature is
   smooth and seasonal; wind speed and direction are volatile. A single modest
   network had to split its capacity between them and did neither well.
2. **Mean squared error pushed predictions toward the average.** Squared error
   penalises large misses so heavily that the safest strategy is to stay near
   the mean -- exactly wrong for a system meant to flag unusual conditions.
3. **No calendar features.** The model was never given month or day-of-year, so
   it had to infer the annual cycle from the data alone.
4. **A single 7-day window for every variable.** Temperature may want a much
   longer window than wind.
5. **Metrics were reported on scaled values**, which makes them hard for
   non-technical stakeholders to interpret.

## Recommended next phase

- Separate models for temperature and for wind, rather than one shared model
- Explicit seasonal and calendar features
- A loss function that does not flatten extremes
- Compare 3, 7, 14 and 30-day windows per variable
- Report error in real units (degrees Celsius, hPa)
- Bring in buoy sensor data, which this phase evaluated but did not use

---

## What I would do differently

Reviewing this work later, two things stand out:

- **The scaler was fitted on the full dataset before splitting.** That leaks
  information from the test period into training. The scaler should be fitted on
  the training portion only, then applied to validation and test.
- **The tuned XGBoost model was never scored on the held-out test set.** Tuning
  identified better parameters, but the final evaluation still used the original
  model, so the reported figures are incomplete.

Neither changes the conclusion, but both are the kind of thing worth catching.

---

## Tools

Python, TensorFlow/Keras, XGBoost, scikit-learn, Pandas, NumPy, Matplotlib,
Seaborn, Jupyter/Colab.

## Credits

Phase 6 was a student team project for the Municipal Innovation Council. The
LSTM was built collaboratively by the team; the XGBoost baseline and its tuning
were my contribution. The wider Smart Beach programme is MIC's, and the final
report is shared here with the project's documentation.
