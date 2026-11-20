"""Contact parameters for a calibrated Silver Badger landing surface."""

import hashlib
import json
from pathlib import Path

from .landing_physics import PROJECT_ROOT
from .landing_safety import CALF_LEGS


def read_landing_mat(path):
    path = PROJECT_ROOT / Path(path)
    blob = path.read_bytes()
    return json.loads(blob), hashlib.sha256(blob).hexdigest()


def apply_landing_mat(xml, terrain_type, config):
    if terrain_type != 'plane':
        raise ValueError('The landing mat requires plane terrain')
    profile, sha256 = read_landing_mat(config['profile'])
    if sha256 != config['sha256']:
        raise ValueError('Landing mat profile SHA256 mismatch')

    floor = xml.find('geom', 'floor')
    floor_pairs = {}
    for pair in xml.contact.pair:
        if pair.geom1 is floor:
            floor_pairs[pair.geom2.name] = pair
        elif pair.geom2 is floor:
            floor_pairs[pair.geom1.name] = pair
    calves = {f'{leg}_calf{part}' for leg in CALF_LEGS for part in ('', '_lower')}
    feet = {f'{leg}_foot' for leg in CALF_LEGS}
    if not (calves | feet) <= floor_pairs.keys():
        raise ValueError('The landing mat requires all four split calves and foot contact pairs')

    for name, pair in floor_pairs.items():
        if name in calves and not config['calf_contact']:
            geom = xml.find('geom', name)
            pair.remove()
            geom.contype = 0
            geom.conaffinity = 0
        else:
            pair.solref = profile['solref']
            pair.solimp = profile['solimp']
