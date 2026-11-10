"""Parts of the workbench.

environments.py  which phantoms / environments can be opened, which checkpoints fit them
frame.py         Frame / SceneInfo / EpisodeLog, geometry helpers (numpy only)
policies.py      Policy and SafetyFilter interfaces, built-in policies, checkpoint loader
sources.py       FrameSource interface; live simulator, replay and view-only sources
instrument.py    instrument window: rest shape and numbers of each instrument
text.py          sidebar HTML/SVG (Chinese labels for the observation terms)
scene.py         viser 3D layers and camera presets
body.py          see-through human body around the BIRO phantom (MakeHuman mesh, assets/)
xray.py          X-ray mode: C-arm camera, background, what shows
plots.py         viser live plots
"""
