"""MuJoCo test scenes for sim2sim.  Obstacles are boxes (exact vertical risers and planar ramps)
except the rough patch, which is a height field.  Terrain geoms are in geom group 0 so that the
height scan ray-casts against terrain only.

Course along +x (y in [-1.5, 1.5]):
  start | 30 deg ramp up, platform, 30 deg ramp down | 6 stairs up (12 x 30 cm), platform, 6 down
        | rough patch +-4 cm | discrete blocks 3-8 cm | flat sprint lane
"""
import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.interpolate import RectBivariateSpline

COURSE_WIDTH = 3.0
STONE = ("0.70 0.64 0.58 1", "0.62 0.56 0.50 1")
GREY = "0.55 0.60 0.66 1"


def _box(parent, name, center, half, rgba, axisangle=None):
    attrs = {"name": name, "type": "box", "pos": " ".join(f"{v:.5f}" for v in center),
             "size": " ".join(f"{v:.5f}" for v in half), "rgba": rgba, "group": "0"}
    if axisangle is not None:
        attrs["axisangle"] = " ".join(f"{v:.6f}" for v in axisangle)
    ET.SubElement(parent, "geom", attrs)


def ramp(parent, name, x0, z0, run, angle_deg, up=True, thick=0.1):
    """Planar ramp whose top surface starts at (x0, z0) and covers a horizontal run; returns (x1, z1)."""
    th = math.radians(angle_deg)
    dz = run * math.tan(th) * (1 if up else -1)
    a = -th if up else th                       # rotation about +y
    normal = (math.sin(a), math.cos(a))         # top-surface normal (x, z) of the rotated box
    xm, zm = x0 + run / 2, z0 + dz / 2
    _box(parent, name, (xm - thick / 2 * normal[0], 0, zm - thick / 2 * normal[1]),
         (run / math.cos(th) / 2 + 0.002, COURSE_WIDTH / 2, thick / 2), GREY, axisangle=(0, 1, 0, a))
    return x0 + run, z0 + dz


def block(parent, name, x0, x1, z_top, rgba=GREY):
    _box(parent, name, ((x0 + x1) / 2, 0, z_top / 2), ((x1 - x0) / 2, COURSE_WIDTH / 2, z_top / 2), rgba)


def rough_field(rng, length, width, res=0.04, cell=0.12, taper=0.3):
    nrow, ncol = int(round(width / res)) + 1, int(round(length / res)) + 1
    cy, cx = int(width / cell) + 2, int(length / cell) + 2
    coarse = rng.uniform(0.0, 1.0, (cy, cx))
    spline = RectBivariateSpline(np.linspace(0, width, cy), np.linspace(0, length, cx), coarse, kx=1, ky=1)
    y, x = np.linspace(0, width, nrow), np.linspace(0, length, ncol)
    data = spline(y, x)
    wy = np.clip(np.minimum(y, width - y) / taper, 0, 1)[:, None]
    wx = np.clip(np.minimum(x, length - x) / taper, 0, 1)[None, :]
    return 0.5 + (data - 0.5) * wy * wx       # 0.5 == ground level after the geom offset


def course_geoms(worldbody, asset, rng):
    sections = {}
    x = 2.0
    s0 = x
    x, z = ramp(worldbody, "ramp_up", x, 0.0, 1.2, 30.0, up=True)
    block(worldbody, "ramp_top", x, x + 1.5, z)
    x += 1.5
    x, z = ramp(worldbody, "ramp_down", x, z, 1.2, 30.0, up=False)
    sections["slope_30deg"] = (s0, x)

    x += 1.6
    s0 = x
    rise, tread, n = 0.12, 0.30, 6
    for k in range(n):
        block(worldbody, f"stair_up_{k}", x + k * tread, x + (k + 1) * tread, (k + 1) * rise, STONE[k % 2])
    x += n * tread
    block(worldbody, "stair_top", x, x + 1.2, n * rise)
    x += 1.2
    for k in range(n - 1):
        block(worldbody, f"stair_down_{k}", x + k * tread, x + (k + 1) * tread, (n - 1 - k) * rise, STONE[k % 2])
    x += n * tread
    sections["stairs_12cm"] = (s0, x)

    x += 1.4
    s0, length = x, 4.0
    data = rough_field(rng, length, COURSE_WIDTH)
    ET.SubElement(asset, "hfield", {"name": "rough", "nrow": str(data.shape[0]), "ncol": str(data.shape[1]),
                                    "size": f"{length / 2} {COURSE_WIDTH / 2} 0.08 0.05"})
    ET.SubElement(worldbody, "geom", {"name": "rough_patch", "type": "hfield", "hfield": "rough", "group": "0",
                                      "pos": f"{s0 + length / 2} 0 -0.04", "rgba": "0.52 0.48 0.40 1"})
    x += length
    sections["rough_4cm"] = (s0, x)

    x += 1.5
    s0 = x
    for k in range(40):
        bx, by = rng.uniform(x + 0.2, x + 3.3), rng.uniform(-1.2, 1.2)
        sx, sy, h = rng.uniform(0.1, 0.25), rng.uniform(0.1, 0.25), rng.uniform(0.03, 0.08)
        _box(worldbody, f"blk_{k}", (bx, by, h / 2), (sx, sy, h / 2), "0.45 0.50 0.56 1")
    x += 3.5
    sections["discrete_8cm"] = (s0, x)
    sections["sprint"] = (x + 1.0, x + 30.0)
    return data, sections


def build_scene(robot_xml_path, mesh_dir, scene="course", seed=0):
    """Returns (xml string, height-field data or None, {section: (x_start, x_end)})."""
    tree = ET.parse(robot_xml_path)
    root = tree.getroot()
    root.find("compiler").set("meshdir", mesh_dir)
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    ET.SubElement(asset, "texture", {"name": "grid", "type": "2d", "builtin": "checker", "width": "512",
                                     "height": "512", "rgb1": "0.80 0.82 0.84", "rgb2": "0.70 0.72 0.75"})
    ET.SubElement(asset, "material", {"name": "grid", "texture": "grid", "texrepeat": "60 30", "reflectance": "0.05"})
    ET.SubElement(asset, "texture", {"name": "sky", "type": "skybox", "builtin": "gradient", "rgb1": "0.75 0.84 0.92",
                                     "rgb2": "0.98 0.98 1.0", "width": "512", "height": "3072"})
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": "1920", "offheight": "1080"})
    ET.SubElement(visual, "quality", {"shadowsize": "4096"})
    ET.SubElement(visual, "headlight", {"ambient": "0.4 0.4 0.4", "diffuse": "0.6 0.6 0.6"})
    worldbody.insert(0, ET.Element("geom", {"name": "floor", "type": "plane", "size": "60 30 0.1",
                                            "material": "grid", "group": "0", "pos": "20 0 0"}))
    worldbody.insert(0, ET.Element("light", {"pos": "0 -2 4", "dir": "0.2 0.4 -1", "directional": "true",
                                             "castshadow": "true"}))
    if scene == "course":
        hdata, sections = course_geoms(worldbody, asset, np.random.default_rng(seed))
    elif scene == "flat":
        hdata, sections = None, {"flat": (-40.0, 80.0)}
    elif scene.startswith("ramp"):            # e.g. ramp30: up ramp, platform, down ramp
        deg = float(scene[4:])
        x, z = ramp(worldbody, "ramp_up", 2.0, 0.0, 1.5, deg, up=True)
        block(worldbody, "ramp_top", x, x + 1.5, z)
        x, z = ramp(worldbody, "ramp_down", x + 1.5, z, 1.5, deg, up=False)
        hdata, sections = None, {f"ramp_up_{deg:g}deg": (2.0, 3.5), f"ramp_down_{deg:g}deg": (5.0, x)}
    elif scene.startswith("stairs"):          # e.g. stairs12: 8 steps up, platform, 8 steps down
        rise, tread, n = float(scene[6:]) / 100.0, 0.30, 8
        x = 2.0
        for k in range(n):
            block(worldbody, f"stair_up_{k}", x + k * tread, x + (k + 1) * tread, (k + 1) * rise, STONE[k % 2])
        x += n * tread
        block(worldbody, "stair_top", x, x + 1.2, n * rise)
        x += 1.2
        for k in range(n - 1):
            block(worldbody, f"stair_down_{k}", x + k * tread, x + (k + 1) * tread, (n - 1 - k) * rise, STONE[k % 2])
        hdata, sections = None, {f"stairs_up_{rise * 100:g}cm": (2.0, 2.0 + n * tread),
                                 f"stairs_down_{rise * 100:g}cm": (x, x + n * tread)}
    else:
        raise ValueError(f"unknown scene {scene}")
    return ET.tostring(root, encoding="unicode"), hdata, sections
