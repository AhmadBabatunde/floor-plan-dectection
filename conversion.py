#!/usr/bin/env python3
"""
Convert polygon annotations to YOLO bounding-box labels.

Designed to handle common polygon JSON schemas seen in floorplan datasets
such as CubiCasa5K (arXiv:1904.01920, zenodo:2613548). The script walks an
annotations root, finds JSON files, extracts polygons (rooms/icons), computes
axis-aligned bounding boxes, normalizes to YOLO format, and writes per-image
label files.

Key features:
- Flexible JSON parsing for different field names (objects/annotations/rooms/icons,
  polygon/vertices/points, label/class/type/category, etc.).
- Supports selecting which entity types to export: rooms, icons, or both.
- Accepts an optional class map file; otherwise builds one dynamically and writes
  classes.txt to the output directory.
- Searches for corresponding image files using hints in the JSON or via
  configurable roots and common extensions.

Usage:
  python3 conversion.py \
    --annotations_root /path/to/annotations \
    --images_root /path/to/images \
    --output_labels /path/to/output/labels \
    --entity both

Notes:
- YOLO labels are written as: "<class_id> <cx> <cy> <w> <h>" with all values
  normalized to [0, 1] relative to the image width/height.
- If an image cannot be found or has no valid objects, an empty label file will
  still be created to comply with YOLO training expectations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# PIL is imported lazily in main() so that `--help` works without Pillow installed


# --------------------------- Data Structures ---------------------------

Point = Tuple[float, float]


@dataclass
class PolygonObject:
    label: str
    points: List[Point]
    entity_type: str  # "room" or "icon" or generic


# --------------------------- Helpers ---------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def try_get(obj: dict, keys: Sequence[str], default=None):
    for key in keys:
        if key in obj:
            return obj[key]
    return default


def normalize_label(name: str) -> str:
    if not isinstance(name, str):
        return str(name)
    return name.strip()


def extract_points_from_any(polygon_like) -> Optional[List[Point]]:
    """Attempt to extract a list of (x, y) points from various representations.

    Accepted forms:
    - [[x, y], [x, y], ...]
    - [(x, y), (x, y), ...]
    - {"x": x, "y": y}, ...
    - {"points": [...]}, {"vertices": [...]}, {"polygon": [...]}
    """
    if polygon_like is None:
        return None

    # Direct list of pairs or dicts
    if isinstance(polygon_like, (list, tuple)):
        points: List[Point] = []
        for item in polygon_like:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    x = float(item[0])
                    y = float(item[1])
                    points.append((x, y))
                except Exception:
                    return None
            elif isinstance(item, dict):
                # Common keys: x, y
                if "x" in item and "y" in item:
                    try:
                        x = float(item["x"])  # type: ignore[arg-type]
                        y = float(item["y"])  # type: ignore[arg-type]
                        points.append((x, y))
                    except Exception:
                        return None
                else:
                    return None
            else:
                return None
        return points if len(points) >= 3 else None

    # Dict that nests the points
    if isinstance(polygon_like, dict):
        for key in ("points", "vertices", "polygon"):
            if key in polygon_like:
                return extract_points_from_any(polygon_like[key])

    return None


def polygon_to_bbox(points: Sequence[Point]) -> Optional[Tuple[float, float, float, float]]:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    if not (x_max > x_min and y_max > y_min):
        return None
    return x_min, y_min, x_max, y_max


def bbox_to_yolo(
    bbox_xyxy: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = bbox_xyxy
    cx = (x_min + x_max) / 2.0 / float(img_w)
    cy = (y_min + y_max) / 2.0 / float(img_h)
    w = (x_max - x_min) / float(img_w)
    h = (y_max - y_min) / float(img_h)
    return cx, cy, w, h


def discover_image_path(
    json_data: dict,
    json_path: Path,
    images_root: Optional[Path],
    image_exts: Sequence[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
) -> Optional[Path]:
    # 1) Hints inside JSON
    image_hint = try_get(
        json_data,
        ("image_filename", "imagePath", "image_file", "image", "filename"),
    )
    if isinstance(image_hint, dict):
        image_hint = try_get(image_hint, ("filename", "path", "name"))
    if isinstance(image_hint, str):
        # If hint is absolute or relative path
        candidate = Path(image_hint)
        if candidate.is_file():
            return candidate
        # Try relative to the JSON directory
        relative_candidate = (json_path.parent / candidate).resolve()
        if relative_candidate.is_file():
            return relative_candidate

    # 2) Same stem as JSON next to it
    stem = json_path.stem
    for ext in image_exts:
        sibling = json_path.with_suffix(ext)
        if sibling.is_file():
            return sibling

    # 3) Same stem under images_root (or common subfolders)
    if images_root is not None:
        # exact under root
        for ext in image_exts:
            candidate = images_root / f"{stem}{ext}"
            if candidate.is_file():
                return candidate

        # try common subfolders under images_root
        common_subdirs = (
            "images",
            "imgs",
            "image",
            "floorplans",
            "png",
            "jpg",
        )
        for sub in common_subdirs:
            for ext in image_exts:
                candidate = images_root / sub / f"{stem}{ext}"
                if candidate.is_file():
                    return candidate

    return None


def parse_polygon_objects(json_data: dict) -> Iterable[PolygonObject]:
    """Yield PolygonObject from a flexible JSON schema.

    Attempts a best-effort parse that supports keys observed across datasets.
    """
    # Potential top-level containers
    containers: List[Tuple[str, List[dict]]] = []

    def as_list_of_dicts(value) -> Optional[List[dict]]:
        if isinstance(value, list) and all(isinstance(v, dict) for v in value):
            return value  # type: ignore[return-value]
        return None

    for key in ("objects", "annotations", "items", "shapes"):
        val = json_data.get(key)
        lod = as_list_of_dicts(val)
        if lod is not None:
            containers.append((key, lod))

    # CubiCasa5K-style buckets
    for key in ("rooms", "icons"):
        val = json_data.get(key)
        lod = as_list_of_dicts(val)
        if lod is not None:
            containers.append((key, lod))

    # If nothing matched, maybe the file directly describes a single item
    if not containers and isinstance(json_data, dict):
        maybe_single = json_data.get("object") or json_data.get("annotation")
        if isinstance(maybe_single, dict):
            containers.append(("objects", [maybe_single]))

    for container_key, items in containers:
        entity_type = "icon" if container_key == "icons" else ("room" if container_key == "rooms" else "object")
        for obj in items:
            # Label fields that may appear
            label = try_get(
                obj,
                (
                    "label",
                    "class",
                    "category",
                    "type",
                    "name",
                    "room_type",
                    "icon_class",
                ),
            )
            if label is None:
                continue
            label_str = normalize_label(label)

            # Polygon extraction: check common fields
            poly_candidate = try_get(obj, ("polygon", "poly", "points", "vertices", "contour"))

            # Some schemas nest geometry deeper
            if poly_candidate is None:
                geometry = try_get(obj, ("geometry", "shape", "mask", "segmentation"))
                if geometry is not None:
                    poly_candidate = try_get(geometry, ("polygon", "points", "vertices"))

            points = extract_points_from_any(poly_candidate)
            if points is None:
                # Fallback: if bbox is present, use it directly
                bbox = try_get(obj, ("bbox", "bounding_box", "bounds"))
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    x_min, y_min, x_max, y_max = bbox
                    points = [(float(x_min), float(y_min)), (float(x_max), float(y_min)), (float(x_max), float(y_max))]
                elif isinstance(bbox, dict):
                    x_min = try_get(bbox, ("x_min", "xmin", "left", "x1", "x"))
                    y_min = try_get(bbox, ("y_min", "ymin", "top", "y1", "y"))
                    x_max = try_get(bbox, ("x_max", "xmax", "right", "x2", "x_end"))
                    y_max = try_get(bbox, ("y_max", "ymax", "bottom", "y2", "y_end"))
                    if None not in (x_min, y_min, x_max, y_max):
                        points = [
                            (float(x_min), float(y_min)),
                            (float(x_max), float(y_min)),
                            (float(x_max), float(y_max)),
                        ]

            if points is None:
                continue

            yield PolygonObject(label=label_str, points=list(points), entity_type=entity_type)


def load_or_build_class_mapping(
    mapping_path: Optional[Path],
    discovered_labels: Iterable[str],
) -> Tuple[Dict[str, int], List[str]]:
    if mapping_path and mapping_path.is_file():
        names: List[str] = []
        with mapping_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                names.append(line)
        label_to_id = {name: idx for idx, name in enumerate(names)}
        return label_to_id, names

    # Build from discovered labels (sorted for determinism)
    unique = sorted({normalize_label(lbl) for lbl in discovered_labels if normalize_label(lbl)})
    label_to_id = {name: idx for idx, name in enumerate(unique)}
    return label_to_id, unique


def write_classes_file(output_dir: Path, names: List[str]) -> None:
    ensure_dir(output_dir)
    with (output_dir / "classes.txt").open("w", encoding="utf-8") as f:
        for name in names:
            f.write(f"{name}\n")


def _parse_svg_points_attr(points_str: str) -> Optional[List[Point]]:
    if not points_str:
        return None
    points: List[Point] = []
    # Points may be separated by spaces or commas; pairs may be space or comma-separated
    # Normalize by splitting on spaces, then splitting pairs on comma
    raw_tokens = points_str.strip().replace("\n", " ").split()
    for token in raw_tokens:
        if "," in token:
            try:
                x_str, y_str = token.split(",", 1)
                points.append((float(x_str), float(y_str)))
            except Exception:
                return None
        else:
            # Handle case where format is "x y x y ..."
            # Defer parsing in a second pass
            points = []
            break
    if points:
        return points if len(points) >= 3 else None
    # Second pass for space-separated coordinate list
    try:
        nums = [float(v) for v in raw_tokens]
        if len(nums) % 2 != 0:
            return None
        for i in range(0, len(nums), 2):
            points.append((nums[i], nums[i + 1]))
        return points if len(points) >= 3 else None
    except Exception:
        return None


def _gather_svg_elements(svg_root) -> List[Tuple[str, dict]]:
    # Returns list of (tag_name, attrib) for supported elements
    elements: List[Tuple[str, dict]] = []
    # xml.etree includes namespace in tag: e.g., '{http://www.w3.org/2000/svg}polygon'
    def strip_ns(tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag

    for elem in svg_root.iter():
        tag = strip_ns(elem.tag)
        if tag in {"polygon", "polyline", "rect", "circle", "ellipse", "path"}:
            elements.append((tag, elem.attrib))
    return elements


def _infer_label_from_attrib(attrib: dict) -> Optional[str]:
    # Check common fields for labels/classes
    for key in ("label", "class", "data-label", "data-class", "inkscape:label"):
        if key in attrib and attrib[key]:
            return normalize_label(attrib[key])
    return None


def parse_cubicasa_svg(svg_path: Path) -> List[PolygonObject]:
    import xml.etree.ElementTree as ET
    try:
        tree = ET.parse(str(svg_path))
        root = tree.getroot()
    except Exception as e:
        raise RuntimeError(f"Failed to parse SVG {svg_path}: {e}")

    objects: List[PolygonObject] = []
    elements = _gather_svg_elements(root)
    # Try to extract label info from element attribs. If missing, leave generic.
    for tag, attrib in elements:
        label = _infer_label_from_attrib(attrib) or "object"
        entity_type = "icon" if ("icon" in label.lower()) else ("room" if ("room" in label.lower()) else "object")
        pts: Optional[List[Point]] = None

        if tag in {"polygon", "polyline"}:
            pts = _parse_svg_points_attr(attrib.get("points", ""))
        elif tag == "rect":
            try:
                x = float(attrib.get("x", 0.0))
                y = float(attrib.get("y", 0.0))
                w = float(attrib.get("width", 0.0))
                h = float(attrib.get("height", 0.0))
                if w > 0 and h > 0:
                    pts = [(x, y), (x + w, y), (x + w, y + h)]
            except Exception:
                pts = None
        elif tag == "circle":
            try:
                cx = float(attrib.get("cx", 0.0))
                cy = float(attrib.get("cy", 0.0))
                r = float(attrib.get("r", 0.0))
                if r > 0:
                    pts = [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r)]
            except Exception:
                pts = None
        elif tag == "ellipse":
            try:
                cx = float(attrib.get("cx", 0.0))
                cy = float(attrib.get("cy", 0.0))
                rx = float(attrib.get("rx", 0.0))
                ry = float(attrib.get("ry", 0.0))
                if rx > 0 and ry > 0:
                    pts = [(cx - rx, cy - ry), (cx + rx, cy - ry), (cx + rx, cy + ry)]
            except Exception:
                pts = None
        elif tag == "path":
            # Optional: if svgpathtools is available, compute bbox; otherwise skip
            try:
                from svgpathtools import parse_path  # type: ignore
                d = attrib.get("d", "")
                if d:
                    p = parse_path(d)
                    xmin, xmax, ymin, ymax = p.bbox()
                    pts = [(xmin, ymin), (xmax, ymin), (xmax, ymax)]
            except Exception:
                pts = None

        if pts is None:
            continue
        objects.append(PolygonObject(label=label, points=pts, entity_type=entity_type))

    return objects


def _read_split_file(path: Path) -> List[str]:
    items: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(line)
    return items


def _resolve_image_path(dataset_root: Path, rel: str, variant: str) -> Path:
    # rel may be a folder or a file path
    rel_path = (dataset_root / rel).resolve()
    if rel_path.is_dir():
        name = "F1_scaled.png" if variant == "scaled" else "F1_original.png"
        return (rel_path / name).resolve()
    # If it points directly to image, return as is
    return rel_path


def convert_cubicasa_svg(
    dataset_root: Path,
    split_name: str,
    split_items: List[str],
    variant: str,
    output_labels_root: Path,
    label_to_id: Dict[str, int],
    quiet: bool = False,
) -> Tuple[int, int]:
    # First pass already computed mapping; here we write labels
    total_images = 0
    total_objects = 0
    try:
        from PIL import Image  # type: ignore
    except Exception:
        print("Pillow (PIL) is required. Install with: pip install Pillow", file=sys.stderr)
        raise

    for item in split_items:
        image_path = _resolve_image_path(dataset_root, item, variant)
        if not image_path.is_file():
            if not quiet:
                print(f"[WARN] Missing image: {image_path}")
            continue
        svg_path = image_path.parent / "model.svg"
        if not svg_path.is_file():
            if not quiet:
                print(f"[WARN] Missing SVG: {svg_path}")
            # Still write empty label file to maintain consistency
            out_dir = output_labels_root / split_name / image_path.parent.relative_to(dataset_root)
            ensure_dir(out_dir)
            with (out_dir / f"{image_path.stem}.txt").open("w", encoding="utf-8"):
                pass
            total_images += 1
            continue

        try:
            objects = parse_cubicasa_svg(svg_path)
        except Exception as e:
            if not quiet:
                print(f"[WARN] Failed to parse {svg_path}: {e}")
            continue

        try:
            with Image.open(str(image_path)) as im:
                img_w, img_h = im.size
        except Exception as e:
            if not quiet:
                print(f"[WARN] Cannot open image {image_path}: {e}")
            continue

        yolo_lines: List[str] = []
        obj_count = 0
        for obj in objects:
            bbox = polygon_to_bbox(obj.points)
            if bbox is None:
                continue
            cx, cy, w, h = bbox_to_yolo(bbox, img_w, img_h)
            if obj.label not in label_to_id:
                # Skip unknown labels (should not happen after discovery)
                continue
            class_id = label_to_id[obj.label]
            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            obj_count += 1

        out_dir = output_labels_root / split_name / image_path.parent.relative_to(dataset_root)
        ensure_dir(out_dir)
        with (out_dir / f"{image_path.stem}.txt").open("w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        total_images += 1
        total_objects += obj_count
        if not quiet:
            print(f"[{split_name}] {image_path.stem}: {obj_count} objects")

    return total_images, total_objects

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert polygon annotations to YOLO labels.")
    parser.add_argument("--annotations_root", type=str, required=True, help="Root directory containing annotation JSON files")
    parser.add_argument("--images_root", type=str, required=False, default=None, help="Root directory containing images (optional)")
    parser.add_argument("--output_labels", type=str, required=True, help="Directory to write YOLO label .txt files")
    parser.add_argument("--class_map", type=str, default=None, help="Optional path to classes.txt mapping file (one class name per line)")
    parser.add_argument(
        "--entity",
        type=str,
        default="both",
        choices=["rooms", "icons", "both"],
        help="Which entities to export to labels",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    parser.add_argument("--cubicasa_root", type=str, default=None, help="If set, enables CubiCasa5K SVG mode using train/val/test split txt files located here")
    parser.add_argument("--split_variant", type=str, default="scaled", choices=["scaled", "original"], help="Which image variant to pair with model.svg when in CubiCasa mode")

    args = parser.parse_args()

    # Lazy import Pillow here to allow --help without dependency
    try:
        from PIL import Image  # type: ignore
    except Exception:
        print("Pillow (PIL) is required. Install with: pip install Pillow", file=sys.stderr)
        raise

    annotations_root = Path(args.annotations_root).resolve()
    images_root = Path(args.images_root).resolve() if args.images_root else None
    output_labels = Path(args.output_labels).resolve()
    class_map_path = Path(args.class_map).resolve() if args.class_map else None
    cubicasa_root = Path(args.cubicasa_root).resolve() if args.cubicasa_root else None
    ensure_dir(output_labels)

    # Discovery pass for all labels to build mapping if needed
    discovered_labels: List[str] = []
    json_paths: List[Path] = []

    if cubicasa_root is not None:
        # Discover labels from SVGs by scanning a subset or all items in split files
        split_files = {name: cubicasa_root / f"{name}.txt" for name in ("train", "val", "test")}
        split_items: Dict[str, List[str]] = {}
        for name, path in split_files.items():
            if path.is_file():
                split_items[name] = _read_split_file(path)
            else:
                split_items[name] = []

        # To build class mapping, scan a subset of items (or all if small)
        scan_list: List[str] = split_items.get("train", []) + split_items.get("val", []) + split_items.get("test", [])
        # Limit scan for speed if extremely large
        max_scan = max(1, min(5000, len(scan_list)))
        scan_list = scan_list[:max_scan]
        for rel in scan_list:
            image_path = _resolve_image_path(cubicasa_root, rel, args.split_variant)
            svg_path = image_path.parent / "model.svg"
            if not svg_path.is_file():
                continue
            try:
                for obj in parse_cubicasa_svg(svg_path):
                    discovered_labels.append(obj.label)
            except Exception:
                continue
    else:
        for root, _dirs, files in os.walk(annotations_root):
            for fname in files:
                if fname.lower().endswith(".json"):
                    json_path = Path(root) / fname
                    json_paths.append(json_path)
                    try:
                        data = read_json(json_path)
                        for obj in parse_polygon_objects(data):
                            if args.entity == "rooms" and obj.entity_type != "room":
                                continue
                            if args.entity == "icons" and obj.entity_type != "icon":
                                continue
                            discovered_labels.append(obj.label)
                    except Exception as e:
                        if not args.quiet:
                            print(f"[WARN] Failed to scan {json_path}: {e}")

    label_to_id, class_names = load_or_build_class_mapping(class_map_path, discovered_labels)
    write_classes_file(output_labels, class_names)
    if not args.quiet:
        print(f"Found {len(class_names)} classes. Writing classes.txt to {output_labels}")

    # Conversion pass
    total_images = 0
    total_objects = 0

    if cubicasa_root is not None:
        split_files = {name: cubicasa_root / f"{name}.txt" for name in ("train", "val", "test")}
        split_items: Dict[str, List[str]] = {}
        for name, path in split_files.items():
            split_items[name] = _read_split_file(path) if path.is_file() else []

        for split_name, items in split_items.items():
            imgs, objs = convert_cubicasa_svg(
                dataset_root=cubicasa_root,
                split_name=split_name,
                split_items=items,
                variant=args.split_variant,
                output_labels_root=output_labels,
                label_to_id=label_to_id,
                quiet=args.quiet,
            )
            total_images += imgs
            total_objects += objs
        if not args.quiet:
            print(f"Done (CubiCasa). Processed {total_images} images with {total_objects} objects. Labels at: {output_labels}")
        return

    for json_path in json_paths:
        try:
            data = read_json(json_path)
        except Exception as e:
            if not args.quiet:
                print(f"[WARN] Skipping {json_path}, cannot read JSON: {e}")
            continue

        image_path = discover_image_path(data, json_path, images_root)
        if image_path is None or not image_path.is_file():
            if not args.quiet:
                print(f"[WARN] No image found for {json_path}")
            # Still write an empty label file using json stem
            label_out = output_labels / f"{json_path.stem}.txt"
            ensure_dir(label_out.parent)
            with label_out.open("w", encoding="utf-8"):
                pass
            continue

        try:
            with Image.open(str(image_path)) as im:
                img_w, img_h = im.size
        except Exception as e:
            if not args.quiet:
                print(f"[WARN] Cannot open image {image_path}: {e}")
            continue

        yolo_lines: List[str] = []
        obj_count = 0
        for obj in parse_polygon_objects(data):
            if args.entity == "rooms" and obj.entity_type != "room":
                continue
            if args.entity == "icons" and obj.entity_type != "icon":
                continue
            bbox = polygon_to_bbox(obj.points)
            if bbox is None:
                continue
            cx, cy, w, h = bbox_to_yolo(bbox, img_w, img_h)
            if obj.label not in label_to_id:
                # If mapping was provided explicitly, skip unknown labels.
                # If mapping was auto-built, this should not happen; but skip to be safe.
                continue
            class_id = label_to_id[obj.label]
            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            obj_count += 1

        # Write label file alongside output_labels with image stem
        label_out = output_labels / f"{image_path.stem}.txt"
        ensure_dir(label_out.parent)
        with label_out.open("w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        total_images += 1
        total_objects += obj_count
        if not args.quiet:
            print(f"Wrote {label_out} with {obj_count} objects")

    if not args.quiet:
        print(f"Done. Processed {total_images} images with {total_objects} objects. Labels at: {output_labels}")


if __name__ == "__main__":
    main()