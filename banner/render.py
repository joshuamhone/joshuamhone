"""Render the animated profile banner as self-contained SVG.

    python banner/render.py                 # writes the wide and compact banners, dark and light, to assets/
    python banner/render.py --distance 3.2  # any Config field can be overridden

The globe is a unit sphere seen through a pinhole camera. Every frame is
computed, nothing is faked with 2D ellipses:

  spin about the polar axis -> tilt towards the camera -> perspective divide

A surface point P is visible when the camera can see its outward normal,
which for a unit sphere centred at the origin with the camera at (0, 0, d) is
d * P.z > 1. The same quantity, normalised, drives line opacity, so lines
fade into the limb and the far hemisphere is culled.

Meridians are spaced `step` degrees apart. Rotating the sphere by one step
maps the set of meridians onto itself, so animating a single step and
repeating it gives a seamless, endless rotation at a fraction of the size of a
full turn. Latitude circles are invariant under the spin, so they are static.

Output is deterministic: no randomness, fixed precision, fixed ordering.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import io
import math
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont

HERE = Path(__file__).resolve().parent
Vec = tuple[float, float, float]


@dataclasses.dataclass(frozen=True)
class Config:
    # Camera
    distance: float = 3.6  # camera distance in sphere radii; lower is stronger perspective
    tilt: float = 18.0  # degrees the north pole leans towards the camera
    # Wireframe
    step: int = 12  # degrees between meridians
    latitudes: tuple[int, ...] = (-72, -54, -36, -18, 0, 18, 36, 54, 72)
    segments: int = 8  # quadratic segments per half meridian
    keyframes: int = 2  # intervals per step; 6 degrees apart, so straight-line interpolation is within 0.2 px
    seconds_per_step: float = 1.5  # one full turn = 360 / step * this = 45 s
    # Depth shading of the wireframe. Front: floor + (peak - floor) * facing ** gamma.
    # Rear: rear_near at the limb, falling to rear_far at the point farthest away.
    peak: float = 0.85
    floor: float = 0.3
    gamma: float = 1.0
    rear_near: float = 0.07
    rear_far: float = 0.02
    # Ocean Wireframe palette
    line_color: str = "#3f93ea"
    geodesic_color: str = "#8fd2ff"
    geodesic_peak: float = 1.0
    geodesic_width: float = 1.6
    rim_color: str = "#6ad4ff"
    fill_core: str = "#01050b"  # sphere interior at the centre
    fill_mid: str = "#031428"
    fill_edge: str = "#0a2c50"  # sphere interior at the limb
    rim_glow: float = 0.1  # intensity of the cyan illumination around the limb, 0 to 1
    halo_width: float = 0.03  # how far the illumination reaches, as a fraction of the radius
    limb_opacity: float = 1.0  # crisp edge line where the light falls, top left
    limb_falloff: float = 0.4  # edge line opacity at the bottom right, relative to the top left
    # Geodesic: a great circle inclined to the equator
    geodesic_inclination: float = 38.0  # angle between its plane and the equator
    geodesic_node: float = -32.0  # longitude where it crosses the equator, seen from the camera
    precision: int = 1


TEXT = {
    "l1": "I exploit",
    "l2": "assumptions.",
    "sub": "Occasionally, I write the code responsible for them.",
}

@dataclasses.dataclass(frozen=True)
class Layout:
    width: int
    height: int
    cx: float
    cy: float
    radius: float  # on-screen radius of the limb, in px
    sizes: dict  # css font sizes per text class
    lines: tuple  # (css class, x, y, text)


LAYOUTS = {
    # Desktop: text left, globe right.
    "": Layout(
        1200, 420, 930, 210, 150,
        {"n": 16, "h": 86, "s": 23, "l": 15},
        (
            ("h", 66, 172, TEXT["l1"]),
            ("h", 66, 260, TEXT["l2"]),
            ("s", 70, 306, TEXT["sub"]),
        ),
    ),
    # Phones: drawn for a ~360 px column, so small text stays legible after scaling.
    "compact": Layout(
        720, 584, 360, 150, 112,
        {"n": 22, "h": 76, "s": 29, "l": 20},
        (
            ("h", 46, 362, TEXT["l1"]),
            ("h", 46, 442, TEXT["l2"]),
            ("s", 50, 496, "Occasionally, I write the code"),
            ("s", 50, 534, "responsible for them."),
        ),
    ),
}

THEMES = {
    # The globe uses the Config palette in both themes; themes only set the page around it.
    "dark": {"bg": "#07090c", "ink": "#eef1f5", "sub": "#9aa5b4", "small": "#8592a3", "edge": "#1b222c"},
    "light": {"bg": "#f6f8fa", "ink": "#0b0f14", "sub": "#4b5563", "small": "#5b6573", "edge": "#d0d7de"},
}


class Camera:
    def __init__(self, cfg: Config, layout: Layout):
        self.cfg, self.layout = cfg, layout
        d = cfg.distance
        # Scale chosen so the sphere's silhouette lands exactly on layout.radius.
        self.scale = layout.radius * math.sqrt(d * d - 1) / d
        t = math.radians(cfg.tilt)
        self.ct, self.st = math.cos(t), math.sin(t)

    def world(self, lat: float, lon: float, spin: float) -> Vec:
        """Point on the unit sphere at (lat, lon), spun by `spin`, then tilted."""
        la, lo = math.radians(lat), math.radians(lon + spin)
        x, y, z = math.cos(la) * math.sin(lo), math.sin(la), math.cos(la) * math.cos(lo)
        return self.tilt((x, y, z))

    def tilt(self, p: Vec) -> Vec:
        x, y, z = p
        return x, y * self.ct - z * self.st, y * self.st + z * self.ct

    def project(self, p: Vec) -> tuple[float, float]:
        x, y, z = p
        k = self.cfg.distance / (self.cfg.distance - z)
        return self.layout.cx + self.scale * x * k, self.layout.cy - self.scale * y * k

    def facing(self, p: Vec) -> float:
        """0 at the limb and beyond, 1 at the point nearest the camera."""
        d = self.cfg.distance
        return max(0.0, (d * p[2] - 1) / (d - 1))

    def opacity(self, p: Vec) -> float:
        """Line opacity from depth: bright facing the camera, faint behind the sphere."""
        c, d = self.cfg, self.cfg.distance
        f = self.facing(p)
        if f > 0:
            return c.floor + (c.peak - c.floor) * f**c.gamma
        rear = min(1.0, max(0.0, (1 - d * p[2]) / (d + 1)))  # 0 at the limb, 1 at the far pole
        return c.rear_near + (c.rear_far - c.rear_near) * rear


def unit(v: Vec) -> Vec:
    n = math.sqrt(sum(a * a for a in v))
    return v[0] / n, v[1] / n, v[2] / n


def mid_on_sphere(a: Vec, b: Vec) -> Vec:
    return unit(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2))


def quad(cam: Camera, a: Vec, b: Vec, fmt) -> str:
    """Quadratic Bezier through the projections of a, the sphere midpoint, and b."""
    m = mid_on_sphere(a, b)
    (ax, ay), (mx, my), (bx, by) = cam.project(a), cam.project(m), cam.project(b)
    qx, qy = 2 * mx - (ax + bx) / 2, 2 * my - (ay + by) / 2
    return f"M{fmt(ax)} {fmt(ay)}Q{fmt(qx)} {fmt(qy)} {fmt(bx)} {fmt(by)}"


def segment_opacity(cam: Camera, a: Vec, b: Vec) -> float:
    return cam.opacity(mid_on_sphere(a, b))


def font_data(name: str, text: str) -> str:
    font = TTFont(HERE / "fonts" / name, recalcTimestamp=False)  # keep output byte-identical
    opts = subset.Options()
    opts.flavor = "woff2"
    opts.layout_features = ["kern"]
    opts.name_IDs = []
    opts.drop_tables += ["DSIG", "FFTM"]
    sub = subset.Subsetter(opts)
    sub.populate(text=text)
    sub.subset(font)
    font.flavor = "woff2"
    buf = io.BytesIO()
    font.save(buf)
    return base64.b64encode(buf.getvalue()).decode()


def render(theme: str, cfg: Config, layout: Layout) -> str:
    c = THEMES[theme]
    cam = Camera(cfg, layout)
    p = cfg.precision
    fmt = lambda v: f"{v:.{p}f}".rstrip("0").rstrip(".") if p else f"{round(v)}"
    op = lambda v: f"{v:.2f}".rstrip("0").rstrip(".") or "0"
    dur = f"{cfg.seconds_per_step:g}s"

    # Meridians: each half meridian is split into fixed segments, so every frame has
    # the same path structure and SMIL can interpolate between keyframes.
    lat_edges = [-90 + 180 * i / cfg.segments for i in range(cfg.segments + 1)]
    frames = [cfg.step * k / cfg.keyframes for k in range(cfg.keyframes + 1)]
    meridians = []
    still: dict[str, list[str]] = {}  # frame 0, grouped by opacity, for reduced motion
    for lon in range(0, 360, cfg.step):
        for i in range(cfg.segments):
            ds, ops = [], []
            for spin in frames:
                a = cam.world(lat_edges[i], lon, spin)
                b = cam.world(lat_edges[i + 1], lon, spin)
                ds.append(quad(cam, a, b, fmt))
                ops.append(op(segment_opacity(cam, a, b)))
            still.setdefault(ops[0], []).append(ds[0])
            meridians.append(
                f'<path d="{ds[0]}" stroke-opacity="{ops[0]}">'
                f'<animate attributeName="d" dur="{dur}" repeatCount="indefinite" values="{";".join(ds)}"/>'
                f'<animate attributeName="stroke-opacity" dur="{dur}" repeatCount="indefinite" values="{";".join(ops)}"/>'
                "</path>"
            )

    # Latitudes are invariant under the spin: static geometry, static per-segment depth.
    by_opacity: dict[str, list[str]] = {}
    for lat in cfg.latitudes:
        n = 24
        for i in range(n):
            a = cam.world(lat, 360 * i / n, 0)
            b = cam.world(lat, 360 * (i + 1) / n, 0)
            by_opacity.setdefault(op(segment_opacity(cam, a, b)), []).append(quad(cam, a, b, fmt))
    latitudes = "".join(f'<path stroke-opacity="{o}" d="{"".join(d)}"/>' for o, d in sorted(by_opacity.items()))
    meridians_still = "".join(f'<path stroke-opacity="{o}" d="{"".join(d)}"/>' for o, d in sorted(still.items()))

    # Geodesic: a great circle inclined to the equator, fixed in space while the
    # globe turns beneath it. Only its near half is drawn.
    inc, node = math.radians(cfg.geodesic_inclination), math.radians(cfg.geodesic_node)
    # Orthonormal basis of the circle's plane: e1 is the ascending node on the
    # equator, e2 is 90 degrees further along the circle, raised by the inclination.
    e1 = (math.sin(node), 0.0, math.cos(node))
    e2 = (math.cos(inc) * math.cos(node), math.sin(inc), -math.cos(inc) * math.sin(node))

    def geo(u: float) -> Vec:
        cu, su = math.cos(u), math.sin(u)
        return cam.tilt((cu * e1[0] + su * e2[0], cu * e1[1] + su * e2[1], cu * e1[2] + su * e2[2]))

    arc = {}
    n = 48
    for i in range(n):
        a, b = geo(2 * math.pi * i / n), geo(2 * math.pi * (i + 1) / n)
        f = cam.facing(mid_on_sphere(a, b))
        if f > 0:  # the orbital line is drawn on the near hemisphere only
            arc.setdefault(op(cfg.geodesic_peak * min(1.0, 0.35 + f * 2.2)), []).append(quad(cam, a, b, fmt))
    geodesic_arc = "".join(f'<path stroke-opacity="{o}" d="{"".join(d)}"/>' for o, d in sorted(arc.items()))

    used = {k: "".join(t for cls, _, _, t in layout.lines if cls in k) for k in ("h", "s", "nl")}
    display = font_data("Inter-SemiBold.woff2", used["h"])
    body = font_data("Inter-Regular.woff2", used["s"])
    # The monospace face is embedded only if a layout draws monospace text.
    mono = font_data("JetBrainsMono-Regular.woff2", used["nl"]) if used["nl"] else None
    z = layout.sizes
    texts = "\n".join(
        f'<text class="{cls} in a{min(i + 1, 5)}" x="{x}" y="{y}">{t.replace("&", "&amp;")}</text>'
        for i, (cls, x, y, t) in enumerate(layout.lines)
    )

    # Sphere body: dark core brightening towards the limb, then a narrow cyan
    # illumination just outside it. Both are radial gradients centred on the sphere.
    R, X, Y = layout.radius, fmt(layout.cx), fmt(layout.cy)
    halo_r = R * (1 + cfg.halo_width)
    inner = R / halo_r
    shading = (
        f'<radialGradient id="body" cx="{X}" cy="{Y}" r="{fmt(R)}" gradientUnits="userSpaceOnUse">'
        f'<stop offset="0" stop-color="{cfg.fill_core}"/><stop offset=".72" stop-color="{cfg.fill_mid}"/>'
        f'<stop offset=".955" stop-color="{cfg.fill_edge}"/><stop offset="1" stop-color="{cfg.rim_color}" stop-opacity="{op(cfg.rim_glow * 2)}"/>'
        "</radialGradient>"
        f'<radialGradient id="halo" cx="{X}" cy="{Y}" r="{fmt(halo_r)}" gradientUnits="userSpaceOnUse">'
        f'<stop offset="{op(inner)}" stop-color="{cfg.rim_color}" stop-opacity="{op(cfg.rim_glow)}"/>'
        f'<stop offset="{op(inner + (1 - inner) * 0.4)}" stop-color="{cfg.rim_color}" stop-opacity="{op(cfg.rim_glow * 0.3)}"/>'
        f'<stop offset="1" stop-color="{cfg.rim_color}" stop-opacity="0"/>'
        "</radialGradient>"
        # The edge line is lit from the top left and fades towards the bottom right.
        f'<linearGradient id="limb" x1="{fmt(layout.cx - R * 0.6)}" y1="{fmt(layout.cy - R * 0.8)}" x2="{fmt(layout.cx + R * 0.6)}" y2="{fmt(layout.cy + R * 0.8)}" gradientUnits="userSpaceOnUse">'
        f'<stop offset="0" stop-color="{cfg.rim_color}" stop-opacity="{op(cfg.limb_opacity)}"/>'
        f'<stop offset="1" stop-color="{cfg.rim_color}" stop-opacity="{op(cfg.limb_opacity * cfg.limb_falloff)}"/>'
        "</linearGradient>"
    )

    W, H = layout.width, layout.height
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t d">
<title id="t">Joshua Mhone</title>
<desc id="d">I exploit assumptions. Occasionally, I write the code responsible for them.</desc>
<style>
@font-face{{font-family:D;font-weight:600;src:url(data:font/woff2;base64,{display}) format("woff2")}}
@font-face{{font-family:B;font-weight:400;src:url(data:font/woff2;base64,{body}) format("woff2")}}
{f'@font-face{{font-family:M;font-weight:400;src:url(data:font/woff2;base64,{mono}) format("woff2")}}' if mono else ''}
.n{{font:400 {z['n']}px M,ui-monospace,monospace;letter-spacing:4.5px;fill:{c['small']}}}
.h{{font:600 {z['h']}px D,Inter,system-ui,sans-serif;letter-spacing:-2.6px;fill:{c['ink']}}}
.s{{font:400 {z['s']}px B,Inter,system-ui,sans-serif;fill:{c['sub']}}}
.l{{font:400 {z['l']}px M,ui-monospace,monospace;letter-spacing:2.6px;fill:{c['small']}}}
.in{{opacity:0;animation:r .9s cubic-bezier(.2,.7,.2,1) forwards}}
.a1{{animation-delay:.1s}}.a2{{animation-delay:.22s}}.a3{{animation-delay:.34s}}.a4{{animation-delay:.55s}}.a5{{animation-delay:.72s}}
.g{{opacity:0;animation:f 1.4s ease .25s forwards}}
@keyframes r{{from{{opacity:0;transform:translateY(12px)}}to{{opacity:1;transform:none}}}}
@keyframes f{{to{{opacity:1}}}}
.st{{display:none}}
@media (prefers-reduced-motion:reduce){{.in,.g{{animation:none;opacity:1}}.m{{display:none}}.st{{display:inline}}}}
</style>
<defs>{shading}</defs>
<rect width="{W}" height="{H}" rx="14" fill="{c['bg']}"/>
<g class="g">
<circle cx="{X}" cy="{Y}" r="{fmt(halo_r)}" fill="url(#halo)"/>
<circle cx="{X}" cy="{Y}" r="{fmt(R)}" fill="url(#body)"/>
<g fill="none" stroke="{cfg.line_color}" stroke-linecap="round" stroke-width="1">
<g>{latitudes}</g>
<g class="m">{"".join(meridians)}</g>
<g class="st">{meridians_still}</g>
</g>
<g fill="none" stroke="{cfg.geodesic_color}" stroke-linecap="round" stroke-width="{cfg.geodesic_width:g}">{geodesic_arc}</g>
<circle cx="{X}" cy="{Y}" r="{fmt(R)}" fill="none" stroke="url(#limb)" stroke-width="1.2"/>
</g>
{texts}
<rect x=".5" y=".5" width="{W - 1}" height="{H - 1}" rx="13.5" fill="none" stroke="{c['edge']}"/>
</svg>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=HERE.parent / "assets")
    for field in dataclasses.fields(Config):
        if field.type in ("int", "float", "str"):
            parser.add_argument(f"--{field.name.replace('_', '-')}", type={"int": int, "float": float, "str": str}[field.type])
    args = parser.parse_args()
    overrides = {f.name: getattr(args, f.name) for f in dataclasses.fields(Config) if getattr(args, f.name, None) is not None}
    cfg = Config(**overrides)
    args.out.mkdir(parents=True, exist_ok=True)
    for variant, layout in LAYOUTS.items():
        for theme in THEMES:
            name = "-".join(filter(None, ["banner", variant, theme])) + ".svg"
            svg = render(theme, cfg, layout)
            (args.out / name).write_text(svg, encoding="utf-8", newline="\n")
            print(f"{name:28} {len(svg.encode()) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
