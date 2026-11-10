"""C-arm: how to pick the imaging angle for the guidewire task on the lab phantom.

`choose_angle.py` scores every angle in C-arm reach by how much the route to each target branch (the
whole way from the aorta) is drawn over another vessel and how much it is shortened, then picks one angle for all 5 targets or one
per target, and writes them as a table (`data/`). `per_target.py` is the training side: with
`env.scene.fluoroscopy.image_rot_zx: per_target` each episode gets its target branch's angle from that
table (README.md, "In training"); otherwise a config's fixed `image_rot_zx` holds for the whole run.
"""
