"""The two-device task (guidewire + hooked catheter; todo_list/two_devices_training.md).

MDP parts, each registered under its class name (guidewire_rl/env/registry.py) and listed in
configs/two_devices_round1.yaml: ``route`` (the route and its ruler), ``phases`` (the referee,
a scene monitor), ``observations``, ``rewards``, ``terminal``, ``truncation``, ``actions`` (the
safety shell), ``labels``, ``targets``, ``wall`` (the devices' clearance to the vessel mesh: a
monitor, a label and a truncation). ``script`` is the scripted controller (WP-B).

Nothing is imported here: the labels need SOFA, and each part is imported by the registry when
a config names it.
"""
