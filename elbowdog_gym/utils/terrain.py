"""Terrain curriculum for ElbowDog.

Rows = difficulty levels (difficulty = row / num_rows), columns = terrain types.  Every tile has
a flat platform in the middle where robots spawn; heights are zero on the tile border so that
neighbouring tiles join without cliffs.
"""
import numpy as np
from isaacgym import terrain_utils

TERRAIN_TYPES = ["flat", "slope_up", "slope_down", "rough_slope", "stairs_up", "stairs_down",
                 "discrete", "rough", "approach_ramp"]


class ElbowDogTerrain:
    def __init__(self, cfg, num_robots):
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.hs = cfg.horizontal_scale
        self.vs = cfg.vertical_scale

        names = list(cfg.terrain_proportions.keys())
        props = np.array([cfg.terrain_proportions[n] for n in names], dtype=float)
        unknown = set(names) - set(TERRAIN_TYPES)
        if unknown:
            raise ValueError(f"unknown terrain types {unknown}")
        # assign columns to types proportionally (at least one column for every non-zero type)
        counts = np.maximum(np.round(props / props.sum() * cfg.num_cols), (props > 0).astype(int))
        while counts.sum() > cfg.num_cols:
            counts[np.argmax(counts)] -= 1
        while counts.sum() < cfg.num_cols:
            counts[np.argmax(props / props.sum() * cfg.num_cols - counts)] += 1
        self.col_types = np.concatenate([[TERRAIN_TYPES.index(n)] * int(c) for n, c in zip(names, counts)])

        self.width_px = int(self.env_width / self.hs)
        self.length_px = int(self.env_length / self.hs)
        self.border = int(cfg.border_size / self.hs)
        self.tot_rows = cfg.num_rows * self.length_px + 2 * self.border
        self.tot_cols = cfg.num_cols * self.width_px + 2 * self.border
        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.rng = np.random.default_rng(cfg.seed if hasattr(cfg, "seed") else 0)

        for j in range(cfg.num_cols):
            for i in range(cfg.num_rows):
                difficulty = i / cfg.num_rows
                tile = self.make_tile(TERRAIN_TYPES[self.col_types[j]], difficulty)
                self.add_tile(tile, i, j)

        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(
                self.height_field_raw, self.hs, self.vs, cfg.slope_treshold)

    # ------------------------------------------------------------------ tiles
    def _cheb_distance(self):
        x = (np.arange(self.length_px) + 0.5) * self.hs - self.env_length / 2
        y = (np.arange(self.width_px) + 0.5) * self.hs - self.env_width / 2
        return np.maximum(np.abs(x)[:, None], np.abs(y)[None, :])

    def make_tile(self, kind, difficulty):
        c = self.cfg
        tile = terrain_utils.SubTerrain("terrain", width=self.length_px, length=self.width_px,
                                        vertical_scale=self.vs, horizontal_scale=self.hs)
        half = min(self.env_length, self.env_width) / 2
        platform = c.platform_size / 2
        d = self._cheb_distance()

        if kind in ("slope_up", "slope_down", "rough_slope"):
            max_deg = c.max_slope_deg if kind != "rough_slope" else c.max_rough_slope_deg
            s = np.tan(np.radians(max_deg * difficulty))
            h = s * (half - np.clip(d, platform, half))
            if kind == "slope_up" or (kind == "rough_slope" and self.rng.random() < 0.5):
                h = -h
            tile.height_field_raw[:] = np.round(h / self.vs).astype(np.int16)
            if kind == "rough_slope":
                amp = c.rough_slope_noise[0] + c.rough_slope_noise[1] * difficulty
                terrain_utils.random_uniform_terrain(tile, min_height=-amp, max_height=amp,
                                                     step=self.vs, downsampled_scale=c.rough_downsample)
        elif kind in ("stairs_up", "stairs_down"):
            step_h = c.step_height[0] + c.step_height[1] * difficulty
            tread = self.rng.uniform(*c.step_width)
            n_steps = np.floor((half - np.clip(d, platform, half)) / tread)
            n_steps = np.minimum(n_steps, np.floor((half - platform) / tread))
            h = n_steps * step_h * (-1 if kind == "stairs_up" else 1)
            tile.height_field_raw[:] = np.round(h / self.vs).astype(np.int16)
        elif kind == "discrete":
            obs_h = c.discrete_height[0] + c.discrete_height[1] * difficulty
            terrain_utils.discrete_obstacles_terrain(tile, obs_h, c.discrete_size[0], c.discrete_size[1],
                                                     c.discrete_count, platform_size=c.platform_size)
        elif kind == "rough":
            amp = c.rough_noise[0] + c.rough_noise[1] * difficulty
            terrain_utils.random_uniform_terrain(tile, min_height=-amp, max_height=amp,
                                                 step=self.vs, downsampled_scale=c.rough_downsample)
            centre = d < platform
            tile.height_field_raw[centre] = 0
        elif kind == "approach_ramp":
            # flat lead-in, planar ramp up, platform, ramp down, flat. Same profile as the MuJoCo ramp:
            # the robot has to walk from flat ground onto the slope, which the pyramid slopes never require.
            angle = np.radians(c.max_slope_deg * difficulty)
            run, lead, plat = 1.5, 2.0, 1.2
            xs = (np.arange(self.length_px) + 0.5) * self.hs
            rise = run * np.tan(angle)
            h = np.zeros_like(xs)
            h = np.where((xs >= lead) & (xs < lead + run), (xs - lead) / run * rise, h)
            h = np.where((xs >= lead + run) & (xs < lead + run + plat), rise, h)
            down = lead + run + plat
            h = np.where((xs >= down) & (xs < down + run), (down + run - xs) / run * rise, h)
            tile.height_field_raw[:] = np.round(h[:, None] / self.vs).astype(np.int16)
            tile.spawn_x = 0.7
        return tile

    def add_tile(self, tile, row, col):
        x0 = self.border + row * self.length_px
        y0 = self.border + col * self.width_px
        self.height_field_raw[x0:x0 + self.length_px, y0:y0 + self.width_px] = tile.height_field_raw
        cy = (col + 0.5) * self.env_width
        if getattr(tile, "spawn_x", None) is None:
            cx = (row + 0.5) * self.env_length
            r = int(0.5 / self.hs)
            mx, my = self.length_px // 2, self.width_px // 2
            z = np.max(tile.height_field_raw[mx - r:mx + r, my - r:my + r]) * self.vs
        else:
            cx = row * self.env_length + tile.spawn_x
            z = 0.0
        self.env_origins[row, col] = [cx, cy, z]
