"""Build the RL model of ElbowDog V5 from the design-package URDF.

Outputs (the design package itself is never modified):
  assets/elbowdog_v5/urdf/elbowdog_v5.urdf   Isaac Gym training model
  assets/elbowdog_v5/mjcf/elbowdog_v5.xml    MuJoCo sim2sim model (same masses / shapes / limits)
  assets/elbowdog_v5/meshes/*.stl            visual meshes (copied)
  assets/elbowdog_v5/model_info.json         summary used by the training and sim2sim code

Changes with respect to the design URDF:
  1. Joint effort / velocity limits from the motor datasheets instead of the 12 N*m / 8 rad/s
     structural-demo caps (see MOTORS / BUS_VOLTAGE).
  2. The rubber foot pad becomes its own link (sphere collision + its BOM mass) so that foot
     contact forces can be measured; the calf inertial is reduced accordingly, which keeps the
     combined calf+foot mass, COM and inertia identical to the design.
  3. The 2.5 mm chassis plates are replaced by solid boxes (deck, core, four deck cleats) so
     that body/ground contacts are detected reliably.
  4. Joint damping / friction and gazebo tags are dropped; joint friction and rotor inertia are
     modelled explicitly in the training environment and in the MuJoCo model.

Usage: python tools/build_rl_model.py [--src path/to/elbow_dog.urdf]
"""
import argparse
import json
import math
import os
import shutil
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SRC = "/home/dash/桌面/ElbowDog_V5_结构优化与三维仿真/v5/simulation/elbow_dog.urdf"
OUT_DIR = os.path.join(ROOT, "assets", "elbowdog_v5")

BUS_VOLTAGE = 24.0  # GO-M8010-6 accepts 12-30 V, so a single 24 V bus feeds all 12 motors
MOTORS = {
    # Unitree GO-M8010-6 datasheet: 23.7 N*m peak, 30 rad/s at 24 V, ratio 6.33
    "GO-M8010-6": {"peak_torque": 23.7, "max_speed": 30.0 * BUS_VOLTAGE / 24.0},
    # Unitree IM6014 (N6014B-12.6): 31.7 N*m peak (torque and speed in the same direction),
    # 54.2 rad/s no-load at 60 V; no-load speed scales with the supply voltage
    "IM6014": {"peak_torque": 31.7, "max_speed": 54.2 * BUS_VOLTAGE / 60.0},
}
JOINT_MOTOR = {"hip": "GO-M8010-6", "thigh": "GO-M8010-6", "calf": "IM6014"}
ARMATURE = 0.01          # reflected rotor inertia [kg*m^2], same value in Isaac Gym and MuJoCo
FOOT_MASS = 0.055        # rubber foot pad, design BOM
FOOT_RADIUS = 0.016
LEGS = ["FL", "FR", "RL", "RR"]

# chassis collision boxes in base_link frame: (name, center xyz, size xyz) [m]
BASE_BOXES = [("deck", (0.0, 0.0, 0.05125), (0.286, 0.190, 0.0105)),
              ("core", (0.0, 0.0, 0.00625), (0.266, 0.112, 0.0795))]
for _sx, _px in (("F", 1), ("R", -1)):
    for _sy, _py in (("L", 1), ("R", -1)):
        BASE_BOXES.append((f"{_sx}{_sy}_cleat", (_px * 0.1257, _py * 0.065, 0.00075), (0.0146, 0.105, 0.1065)))


def vec(text, n=3):
    vals = [float(v) for v in (text or "").split()]
    return np.array(vals if vals else [0.0] * n)


def fmt(values):
    return " ".join(f"{v:.10g}" for v in values)


def rpy_to_quat(rpy):
    """URDF fixed-axis roll-pitch-yaw -> MuJoCo quaternion (w x y z)."""
    r, p, y = rpy
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]


def inertia_matrix(el):
    g = {k: float(el.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")}
    return np.array([[g["ixx"], g["ixy"], g["ixz"]],
                     [g["ixy"], g["iyy"], g["iyz"]],
                     [g["ixz"], g["iyz"], g["izz"]]])


def shift(inertia, mass, r):
    """Parallel-axis term: inertia about a point displaced by -r from the COM."""
    r = np.asarray(r)
    return inertia + mass * (np.dot(r, r) * np.eye(3) - np.outer(r, r))


def check_inertia(name, inertia):
    eig = np.linalg.eigvalsh(inertia)
    a, b, c = sorted(eig)
    if a <= 0 or a + b < c * (1 - 1e-9):
        raise ValueError(f"{name}: inertia not physical, eigenvalues {eig}")


def set_inertial(link, mass, com, inertia):
    for old in link.findall("inertial"):
        link.remove(old)
    inertial = ET.Element("inertial")
    ET.SubElement(inertial, "origin", xyz=fmt(com), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{mass:.10g}")
    ET.SubElement(inertial, "inertia", ixx=f"{inertia[0, 0]:.10g}", iyy=f"{inertia[1, 1]:.10g}",
                  izz=f"{inertia[2, 2]:.10g}", ixy=f"{inertia[0, 1]:.10g}",
                  ixz=f"{inertia[0, 2]:.10g}", iyz=f"{inertia[1, 2]:.10g}")
    link.insert(0, inertial)


def indent(elem, level=0):
    pad = "\n" + "  " * level
    if len(elem):
        if not (elem.text or "").strip():
            elem.text = pad + "  "
        for child in elem:
            indent(child, level + 1)
        if not (child.tail or "").strip():
            child.tail = pad
    if level and not (elem.tail or "").strip():
        elem.tail = pad


def add_box(link, name, center, size):
    col = ET.SubElement(link, "collision", name=name)
    ET.SubElement(col, "origin", xyz=fmt(center), rpy="0 0 0")
    ET.SubElement(ET.SubElement(col, "geometry"), "box", size=fmt(size))


def build_urdf(src):
    tree = ET.parse(src)
    robot = tree.getroot()
    robot.set("name", "elbowdog_v5")
    links = {l.get("name"): l for l in robot.findall("link")}
    joints = {j.get("name"): j for j in robot.findall("joint")}

    for g in robot.findall("gazebo"):
        robot.remove(g)

    for name, joint in joints.items():
        for d in joint.findall("dynamics"):
            joint.remove(d)
        if joint.get("type") == "revolute":
            motor = MOTORS[JOINT_MOTOR[name.split("_")[1]]]
            lim = joint.find("limit")
            lim.set("effort", f"{motor['peak_torque']:.4g}")
            lim.set("velocity", f"{motor['max_speed']:.4g}")
        elif joint.get("type") == "fixed" and name.endswith("_foot_fixed"):
            joint.set("dont_collapse", "true")

    base = links["base_link"]
    cylinders = []
    for col in base.findall("collision"):
        if col.find("geometry/cylinder") is not None:
            cylinders.append(col)
        base.remove(col)
    for name, center, size in BASE_BOXES:
        add_box(base, f"base_{name}", center, size)
    for i, col in enumerate(cylinders):
        col.set("name", f"base_motor_{i}")
        base.append(col)

    foot_info = {}
    for leg in LEGS:
        calf, foot = links[f"{leg}_calf"], links[f"{leg}_foot"]
        foot_pos = vec(joints[f"{leg}_foot_fixed"].find("origin").get("xyz"))
        for col in calf.findall("collision"):
            if col.find("geometry/sphere") is not None:
                calf.remove(col)
        inertial = calf.find("inertial")
        m_tot = float(inertial.find("mass").get("value"))
        c_tot = vec(inertial.find("origin").get("xyz"))
        i_tot = inertia_matrix(inertial.find("inertia"))
        i_foot = 0.4 * FOOT_MASS * FOOT_RADIUS ** 2 * np.eye(3)
        m_calf = m_tot - FOOT_MASS
        c_calf = (m_tot * c_tot - FOOT_MASS * foot_pos) / m_calf
        i_origin = shift(i_tot, m_tot, c_tot) - shift(i_foot, FOOT_MASS, foot_pos)
        i_calf = i_origin - m_calf * (np.dot(c_calf, c_calf) * np.eye(3) - np.outer(c_calf, c_calf))
        check_inertia(f"{leg}_calf", i_calf)
        set_inertial(calf, m_calf, c_calf, i_calf)
        set_inertial(foot, FOOT_MASS, np.zeros(3), i_foot)
        col = ET.SubElement(foot, "collision", name=f"{leg}_foot_collision")
        ET.SubElement(col, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(ET.SubElement(col, "geometry"), "sphere", radius=f"{FOOT_RADIUS}")
        foot_info[leg] = foot_pos.tolist()

    for link in robot.findall("link"):
        for mesh in link.iter("mesh"):
            mesh.set("filename", "../meshes/" + os.path.basename(mesh.get("filename")))
        inertial = link.find("inertial")
        if inertial is not None:
            check_inertia(link.get("name"), inertia_matrix(inertial.find("inertia")))
    for el in robot.iter():
        el.text = el.text if (el.text or "").strip() else None
        el.tail = None
    indent(robot)
    return tree, foot_info


def collect_meshes(tree):
    return sorted({os.path.basename(m.get("filename")) for m in tree.getroot().iter("mesh")})


def link_mjcf(link, joint, children, parent_of, joints_by_child, out, depth):
    pad = "  " * depth
    name = link.get("name")
    pos = vec(joint.find("origin").get("xyz")) if joint is not None else np.zeros(3)
    extra = ' pos="0 0 0.32"' if joint is None else f' pos="{fmt(pos)}"'
    out.append(f'{pad}<body name="{name}"{extra}>')
    if joint is None:
        out.append(f'{pad}  <freejoint name="floating_base"/>')
        out.append(f'{pad}  <site name="imu" pos="0 0 0.04" size="0.008"/>')
    elif joint.get("type") == "revolute":
        lim = joint.find("limit")
        out.append(f'{pad}  <joint name="{joint.get("name")}" axis="{joint.find("axis").get("xyz")}" '
                   f'range="{lim.get("lower")} {lim.get("upper")}"/>')
    inertial = link.find("inertial")
    if inertial is not None:
        i = inertia_matrix(inertial.find("inertia"))
        out.append(f'{pad}  <inertial pos="{inertial.find("origin").get("xyz")}" '
                   f'mass="{inertial.find("mass").get("value")}" '
                   f'fullinertia="{fmt([i[0, 0], i[1, 1], i[2, 2], i[0, 1], i[0, 2], i[1, 2]])}"/>')
    for k, vis in enumerate(link.findall("visual")):
        org = vis.find("origin")
        mesh = os.path.splitext(os.path.basename(vis.find("geometry/mesh").get("filename")))[0]
        rgba = vis.find("material/color").get("rgba")
        out.append(f'{pad}  <geom name="{name}_vis{k}" class="visual" type="mesh" mesh="{mesh}" pos="{org.get("xyz")}" '
                   f'quat="{fmt(rpy_to_quat(vec(org.get("rpy"))))}" rgba="{rgba}"/>')
    for col in link.findall("collision"):
        org = col.find("origin")
        geo = col.find("geometry")[0]
        attrs = f'name="{col.get("name")}" class="collision" pos="{org.get("xyz")}" quat="{fmt(rpy_to_quat(vec(org.get("rpy"))))}"'
        if geo.tag == "box":
            attrs += f' type="box" size="{fmt(vec(geo.get("size")) / 2)}"'
        elif geo.tag == "cylinder":
            attrs += f' type="cylinder" size="{geo.get("radius")} {float(geo.get("length")) / 2:.10g}"'
        elif geo.tag == "sphere":
            attrs += f' type="sphere" size="{geo.get("radius")}"'
        if name.endswith("_foot"):
            attrs = attrs.replace('class="collision"', 'class="foot"')
        out.append(f'{pad}  <geom {attrs}/>')
    for child in children.get(name, []):
        link_mjcf(child, joints_by_child[child.get("name")], children, parent_of, joints_by_child, out, depth + 1)
    out.append(f'{pad}</body>')


def build_mjcf(tree, meshes):
    robot = tree.getroot()
    links = {l.get("name"): l for l in robot.findall("link")}
    joints_by_child = {j.find("child").get("link"): j for j in robot.findall("joint")}
    children, parent_of = {}, {}
    for j in robot.findall("joint"):
        p, c = j.find("parent").get("link"), j.find("child").get("link")
        children.setdefault(p, []).append(links[c])
        parent_of[c] = p
    body = []
    link_mjcf(links["base_link"], None, children, parent_of, joints_by_child, body, 2)
    actuators = []
    for j in robot.findall("joint"):
        if j.get("type") == "revolute":
            eff = j.find("limit").get("effort")
            actuators.append(f'    <motor name="{j.get("name")}" joint="{j.get("name")}" ctrlrange="-{eff} {eff}"/>')
    mesh_assets = "\n".join(f'    <mesh name="{os.path.splitext(m)[0]}" file="{m}" scale="0.001 0.001 0.001"/>' for m in meshes)
    return f"""<mujoco model="elbowdog_v5">
  <!-- generated by tools/build_rl_model.py from the RL URDF; edit the generator, not this file -->
  <compiler angle="radian" meshdir="../meshes" inertiafromgeom="false" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <default>
    <joint armature="{ARMATURE}" damping="0.0" frictionloss="0.1"/>
    <geom contype="1" conaffinity="1" condim="3" friction="1.0 0.005 0.0001"/>
    <default class="visual">
      <geom contype="0" conaffinity="0" group="2" density="0"/>
    </default>
    <default class="collision">
      <geom group="3" rgba="0.1 0.6 0.2 0.3" density="0"/>
    </default>
    <default class="foot">
      <geom group="3" rgba="0.9 0.3 0.1 0.6" density="0" friction="1.0 0.005 0.0001"/>
    </default>
  </default>
  <asset>
{mesh_assets}
  </asset>
  <worldbody>
{chr(10).join(body)}
  </worldbody>
  <actuator>
{chr(10).join(actuators)}
  </actuator>
  <sensor>
    <framequat name="imu_quat" objtype="site" objname="imu"/>
    <gyro name="imu_gyro" site="imu"/>
    <accelerometer name="imu_acc" site="imu"/>
  </sensor>
</mujoco>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=DEFAULT_SRC)
    args = parser.parse_args()
    src_mesh_dir = os.path.join(os.path.dirname(args.src), "meshes")

    tree, foot_info = build_urdf(args.src)
    meshes = collect_meshes(tree)
    for sub in ("urdf", "mjcf", "meshes"):
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)
    for m in meshes:
        shutil.copy2(os.path.join(src_mesh_dir, m), os.path.join(OUT_DIR, "meshes", m))
    urdf_path = os.path.join(OUT_DIR, "urdf", "elbowdog_v5.urdf")
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    mjcf_path = os.path.join(OUT_DIR, "mjcf", "elbowdog_v5.xml")
    with open(mjcf_path, "w", encoding="utf-8") as f:
        f.write(build_mjcf(tree, meshes))

    robot = tree.getroot()
    masses = {l.get("name"): float(l.find("inertial/mass").get("value"))
              for l in robot.findall("link") if l.find("inertial") is not None}
    joints = []
    for j in robot.findall("joint"):
        if j.get("type") != "revolute":
            continue
        lim = j.find("limit")
        joints.append({"name": j.get("name"), "motor": JOINT_MOTOR[j.get("name").split("_")[1]],
                       "lower": float(lim.get("lower")), "upper": float(lim.get("upper")),
                       "effort": float(lim.get("effort")), "velocity": float(lim.get("velocity"))})
    info = {"source_urdf": os.path.abspath(args.src), "bus_voltage": BUS_VOLTAGE, "motors": MOTORS,
            "armature": ARMATURE, "foot_mass": FOOT_MASS, "foot_radius": FOOT_RADIUS,
            "foot_offset_in_calf": foot_info, "total_mass": sum(masses.values()),
            "link_masses": masses, "joints": joints,
            "base_collision_boxes": [{"name": n, "center": c, "size": s} for n, c, s in BASE_BOXES]}
    with open(os.path.join(OUT_DIR, "model_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"URDF : {urdf_path}\nMJCF : {mjcf_path}\nmass : {info['total_mass']:.6f} kg")
    for j in joints[:3]:
        print(f"  {j['name']:16s} {j['motor']:11s} effort {j['effort']:5.1f} N*m  velocity {j['velocity']:5.2f} rad/s"
              f"  range [{j['lower']}, {j['upper']}]")


if __name__ == "__main__":
    main()
