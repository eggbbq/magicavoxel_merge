"""Offline glTF/GLB structure inspection; not a complete glTF validator.

No Blender, network access or third-party packages required. Does not verify
binary accessor contents, texture appearance or evaluated skin deformation.
"""

import argparse
import json
from pathlib import Path
import struct
import sys


def read_document(path):
    raw = path.read_bytes()
    if raw[:4] != b"glTF":
        return json.loads(raw.decode("utf-8-sig"))
    if len(raw) < 20:
        raise ValueError("Truncated GLB header")
    _, version, total = struct.unpack_from("<4sII", raw)
    if version != 2 or total != len(raw):
        raise ValueError("Invalid GLB version or declared file length")
    position, document = 12, None
    while position < total:
        if position + 8 > total:
            raise ValueError("Truncated chunk header")
        size, kind = struct.unpack_from("<II", raw, position)
        position += 8
        if size % 4 or position + size > total:
            raise ValueError("Invalid GLB chunk length")
        if kind == 0x4E4F534A:
            if document is not None:
                raise ValueError("Multiple JSON chunks")
            document = json.loads(raw[position:position + size].decode("utf-8"))
        position += size
    if document is None:
        raise ValueError("Missing JSON chunk")
    return document


def inspect(document, expected=None, character_only=False):
    nodes = document.get("nodes", [])
    meshes = document.get("meshes", [])
    skins = document.get("skins", [])
    accessors = document.get("accessors", [])
    animations = document.get("animations", [])
    errors, warnings, parents, adjacency = [], [], {}, {}

    def valid(index, items):
        return type(index) is int and 0 <= index < len(items)

    if document.get("asset", {}).get("version") != "2.0":
        errors.append("Expected glTF asset version 2.0")
    for parent, node in enumerate(nodes):
        children = []
        for child in node.get("children", []):
            if not valid(child, nodes):
                errors.append(f"Node {parent}: invalid child {child}")
                continue
            if child in parents:
                errors.append(f"Node {child}: multiple parent references ({parents[child]}, {parent})")
            parents[child] = parent
            children.append(child)
        adjacency[parent] = children
        for field, items in (("mesh", meshes), ("skin", skins)):
            if field in node and not valid(node[field], items):
                errors.append(f"Node {parent}: invalid {field} index")
    # Check every node, including old wrappers outside the active scene.
    state = {}
    for start in range(len(nodes)):
        if state.get(start) == 2:
            continue
        stack = [(start, False)]
        while stack:
            current, leaving = stack.pop()
            if leaving:
                state[current] = 2
            elif state.get(current) == 1:
                errors.append(f"Cycle through node {current}")
            elif state.get(current) != 2:
                state[current] = 1
                stack.append((current, True))
                stack.extend((child, False) for child in reversed(adjacency[current]))

    reachable = set()
    for scene in document.get("scenes", []):
        stack = list(scene.get("nodes", []))
        while stack:
            node = stack.pop()
            if not valid(node, nodes):
                errors.append(f"Invalid scene root {node}")
            elif node not in reachable:
                reachable.add(node)
                stack.extend(adjacency[node])
        if any(root in parents for root in scene.get("nodes", [])):
            errors.append("A scene root also has a parent")
    if len(reachable) != len(nodes):
        warnings.append(f"{len(nodes) - len(reachable)} nodes unreachable from any scene")

    joints = set()
    for index, skin in enumerate(skins):
        for joint in skin.get("joints", []):
            if not valid(joint, nodes):
                errors.append(f"Skin {index}: invalid joint {joint}")
            else:
                joints.add(joint)
        if len(set(skin.get("joints", []))) != len(skin.get("joints", [])):
            errors.append(f"Skin {index}: duplicate joint")
        if "inverseBindMatrices" in skin and not valid(skin["inverseBindMatrices"], accessors):
            errors.append(f"Skin {index}: invalid inverse bind accessor")
    for index, node in enumerate(nodes):
        if "skin" in node and valid(node.get("mesh"), meshes):
            for primitive in meshes[node["mesh"]].get("primitives", []):
                attributes = primitive.get("attributes", {})
                if not {"JOINTS_0", "WEIGHTS_0"} <= attributes.keys():
                    errors.append(f"Skinned mesh node {index}: missing joint/weight attributes")

    clips = []
    for index, animation in enumerate(animations):
        targets, starts, ends = set(), [], []
        samplers = animation.get("samplers", [])
        for channel in animation.get("channels", []):
            target = channel.get("target", {}).get("node")
            if not valid(target, nodes):
                errors.append(f"Animation {index}: missing or invalid node target (pointer animations unsupported by this inspector)")
            else:
                targets.add(target)
            if not valid(channel.get("sampler"), samplers):
                errors.append(f"Animation {index}: invalid sampler")
        for sampler in samplers:
            for kind in ("input", "output"):
                if not valid(sampler.get(kind), accessors):
                    errors.append(f"Animation {index}: invalid {kind} accessor")
            if valid(sampler.get("input"), accessors):
                accessor = accessors[sampler["input"]]
                if accessor.get("min") and accessor.get("max"):
                    starts.append(accessor["min"][0])
                    ends.append(accessor["max"][0])
        clips.append({"name": animation.get("name", f"animation_{index}"),
                      "duration_from_metadata": max(ends) - min(starts) if ends else None,
                      "target_nodes": sorted(targets), "channels": len(animation.get("channels", []))})
    if expected is not None and len(animations) != expected:
        errors.append(f"Expected {expected} animations; found {len(animations)}")
    if character_only:
        for index, node in enumerate(nodes):
            if "camera" in node or "KHR_lights_punctual" in node.get("extensions", {}):
                errors.append(f"Character file contains camera/light node {index}")
            if index not in joints and "mesh" not in node and not adjacency[index]:
                warnings.append(f"Possible unused leaf node: {index} ({node.get('name', '')})")
    return {"nodes": len(nodes), "meshes": len(meshes), "materials": len(document.get("materials", [])),
            "skins": len(skins), "unique_joint_nodes": len(joints), "animations": clips,
            "errors": errors, "warnings": warnings,
            "scope": "Structure and declared metadata only; no binary, deformation or engine validation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--expect-animations", type=int)
    parser.add_argument("--character-only", action="store_true")
    args = parser.parse_args()
    try:
        report = inspect(read_document(args.file), args.expect_animations, args.character_only)
    except (OSError, ValueError, TypeError, KeyError, struct.error) as error:
        print(json.dumps({"errors": [str(error)]}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
