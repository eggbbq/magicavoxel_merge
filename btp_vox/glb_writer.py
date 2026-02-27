"""Wrapper utilities for writing glTF/GLB scenes."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .glb_internal import write_glb_scene


def write_meshes_gltf(
    output_gltf: str | Path,
    *,
    meshes: Iterable[dict[str, np.ndarray]],
    texture_png: bytes | None,
    texture_path: str | None,
    name_prefix: str | None = None,
    alpha_mode: str | None = None,
    alpha_cutoff: float | None = None,
) -> None:
    from pygltflib import (
        GLTF2,
        Accessor,
        Asset,
        Attributes,
        Buffer,
        BufferView,
        Image,
        Material,
        Mesh,
        Node,
        PbrMetallicRoughness,
        Primitive,
        Sampler,
        Scene,
        Texture,
        TextureInfo,
    )
    import math

    if (texture_png is None) == (texture_path is None):
        raise ValueError("Provide exactly one of texture_png or texture_path")

    output_gltf = Path(output_gltf)
    out_dir = output_gltf.parent
    bin_name = f"{output_gltf.stem}.bin"
    bin_path = out_dir / bin_name

    if texture_png is not None:
        tex_name = f"{output_gltf.stem}.png" if name_prefix is None else f"{name_prefix}.png"
        (out_dir / tex_name).write_bytes(texture_png)
        texture_path = tex_name

    ARRAY_BUFFER = 34962
    ELEMENT_ARRAY_BUFFER = 34963
    UNSIGNED_INT = 5125
    FLOAT = 5126
    ACCESSOR_TYPE_SCALAR = "SCALAR"
    ACCESSOR_TYPE_VEC3 = "VEC3"
    ACCESSOR_TYPE_VEC2 = "VEC2"

    def _align4(n: int) -> int:
        return int(math.ceil(n / 4.0) * 4)

    blob = bytearray()
    buffer_views: list[BufferView] = []
    accessors: list[Accessor] = []
    gltf_meshes: list[Mesh] = []
    nodes: list[Node] = []

    def add_view(data: bytes, target: int | None) -> int:
        offset = len(blob)
        blob.extend(data)
        padded = _align4(len(blob))
        if padded != len(blob):
            blob.extend(b"\x00" * (padded - len(blob)))
        view_index = len(buffer_views)
        buffer_views.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(data), target=target))
        return view_index

    meshes_list = list(meshes)
    for m in meshes_list:
        name = m.get("name")
        translation = m.get("translation")
        rotation = m.get("rotation")
        positions = np.asarray(m["positions"], dtype=np.float32)
        normals = np.asarray(m["normals"], dtype=np.float32)
        texcoords = np.asarray(m["texcoords"], dtype=np.float32)
        texcoords1 = m.get("texcoords1")
        if texcoords1 is not None:
            texcoords1 = np.asarray(texcoords1, dtype=np.float32)
        indices = np.asarray(m["indices"], dtype=np.uint32)

        pos_view = add_view(positions.tobytes(), ARRAY_BUFFER)
        pos_accessor_index = len(accessors)
        accessors.append(
            Accessor(
                bufferView=pos_view,
                componentType=FLOAT,
                count=int(positions.shape[0]),
                type=ACCESSOR_TYPE_VEC3,
                min=positions.min(axis=0).tolist(),
                max=positions.max(axis=0).tolist(),
            )
        )

        nrm_view = add_view(normals.tobytes(), ARRAY_BUFFER)
        nrm_accessor_index = len(accessors)
        accessors.append(Accessor(bufferView=nrm_view, componentType=FLOAT, count=int(normals.shape[0]), type=ACCESSOR_TYPE_VEC3))

        uv_view = add_view(texcoords.tobytes(), ARRAY_BUFFER)
        uv_accessor_index = len(accessors)
        accessors.append(Accessor(bufferView=uv_view, componentType=FLOAT, count=int(texcoords.shape[0]), type=ACCESSOR_TYPE_VEC2))

        uv1_accessor_index = None
        if texcoords1 is not None:
            uv1_view = add_view(texcoords1.tobytes(), ARRAY_BUFFER)
            uv1_accessor_index = len(accessors)
            accessors.append(Accessor(bufferView=uv1_view, componentType=FLOAT, count=int(texcoords1.shape[0]), type=ACCESSOR_TYPE_VEC2))

        idx_view = add_view(indices.tobytes(), ELEMENT_ARRAY_BUFFER)
        idx_accessor_index = len(accessors)
        accessors.append(Accessor(bufferView=idx_view, componentType=UNSIGNED_INT, count=int(indices.shape[0]), type=ACCESSOR_TYPE_SCALAR))

        attrs = Attributes(POSITION=pos_accessor_index, NORMAL=nrm_accessor_index, TEXCOORD_0=uv_accessor_index)
        if uv1_accessor_index is not None:
            attrs.TEXCOORD_1 = uv1_accessor_index
        prim = Primitive(attributes=attrs, indices=idx_accessor_index, material=0)
        mesh_index = len(gltf_meshes)
        gltf_meshes.append(Mesh(primitives=[prim], name=name))

        node = Node(mesh=mesh_index, name=name)
        if translation is not None:
            node.translation = [float(translation[0]), float(translation[1]), float(translation[2])]
        if rotation is not None:
            node.rotation = [float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3])]
        nodes.append(node)

    gltf = GLTF2(
        asset=Asset(version="2.0"),
        buffers=[Buffer(byteLength=0, uri=bin_name)],
        bufferViews=buffer_views,
        accessors=accessors,
        meshes=gltf_meshes,
        images=[Image(uri=str(texture_path), name=(f"{name_prefix}_tex" if name_prefix else None))],
        samplers=[Sampler(magFilter=9728, minFilter=9728, wrapS=33071, wrapT=33071)],
        textures=[Texture(sampler=0, source=0, name=(f"{name_prefix}_texture" if name_prefix else None))],
        materials=[
            Material(
                name=(f"{name_prefix}_mat" if name_prefix else None),
                pbrMetallicRoughness=PbrMetallicRoughness(
                    baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                    baseColorTexture=TextureInfo(index=0),
                    metallicFactor=0.0,
                    roughnessFactor=1.0,
                ),
                alphaMode=(str(alpha_mode) if alpha_mode else None),
                alphaCutoff=(float(alpha_cutoff) if alpha_cutoff is not None else None),
                doubleSided=True,
            )
        ],
        nodes=nodes,
        scenes=[Scene(nodes=list(range(len(nodes))))],
        scene=0,
    )

    gltf.buffers[0].byteLength = len(blob)
    bin_path.write_bytes(bytes(blob))
    gltf.save(str(output_gltf))


def write_scene(
    output_glb: str | Path,
    *,
    meshes: Iterable[dict[str, np.ndarray]],
    nodes: list[dict],
    root_node_ids: list[int],
    texture_png: bytes | None,
    texture_path: str | None,
    name_prefix: str | None = None,
    alpha_mode: str | None = None,
    alpha_cutoff: float | None = None,
) -> None:
    """Write a node hierarchy to GLB.

    nodes: list of dicts with optional keys:
      - name: str
      - mesh: int (index into meshes list)
      - children: list[int] (node indices)
      - translation: (x,y,z)
      - rotation: (x,y,z,w)

    This is a minimal writer built on top of magicavoxel_merge.glb's logic.
    """

    # write_glb_scene already supports node translation/rotation per mesh, but doesn't support hierarchy.
    # For now we reuse it by flattening: we create one mesh per node that has mesh attached.
    # NOTE: This function expects `nodes[*].mesh` indices to correspond to `meshes` order.
    from pygltflib import GLTF2, Accessor, Asset, Attributes, Buffer, BufferView, Image, Material, Mesh, Node, PbrMetallicRoughness, Primitive, Sampler, Scene, Texture, TextureInfo
    import math

    if (texture_png is None) == (texture_path is None):
        raise ValueError("Provide exactly one of texture_png or texture_path")

    ARRAY_BUFFER = 34962
    ELEMENT_ARRAY_BUFFER = 34963
    UNSIGNED_INT = 5125
    FLOAT = 5126
    ACCESSOR_TYPE_SCALAR = "SCALAR"
    ACCESSOR_TYPE_VEC3 = "VEC3"
    ACCESSOR_TYPE_VEC2 = "VEC2"

    def _align4(n: int) -> int:
        return int(math.ceil(n / 4.0) * 4)

    blob = bytearray()

    buffer_views: list[BufferView] = []
    accessors: list[Accessor] = []
    gltf_meshes: list[Mesh] = []
    gltf_nodes: list[Node] = []

    def add_view(data: bytes, target: int | None) -> int:
        offset = len(blob)
        blob.extend(data)
        padded = _align4(len(blob))
        if padded != len(blob):
            blob.extend(b"\x00" * (padded - len(blob)))
        view_index = len(buffer_views)
        buffer_views.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(data), target=target))
        return view_index

    img_view: int | None = None
    if texture_png is not None:
        img_view = add_view(texture_png, None)

    meshes_list = list(meshes)
    for m in meshes_list:
        name = m.get("name")
        positions = np.asarray(m["positions"], dtype=np.float32)
        normals = np.asarray(m["normals"], dtype=np.float32)
        texcoords = np.asarray(m["texcoords"], dtype=np.float32)
        texcoords1 = m.get("texcoords1")
        if texcoords1 is not None:
            texcoords1 = np.asarray(texcoords1, dtype=np.float32)
        indices = np.asarray(m["indices"], dtype=np.uint32)

        pos_view = add_view(positions.tobytes(), ARRAY_BUFFER)
        pos_accessor_index = len(accessors)
        accessors.append(
            Accessor(
                bufferView=pos_view,
                componentType=FLOAT,
                count=int(positions.shape[0]),
                type=ACCESSOR_TYPE_VEC3,
                min=positions.min(axis=0).tolist(),
                max=positions.max(axis=0).tolist(),
            )
        )

        nrm_view = add_view(normals.tobytes(), ARRAY_BUFFER)
        nrm_accessor_index = len(accessors)
        accessors.append(Accessor(bufferView=nrm_view, componentType=FLOAT, count=int(normals.shape[0]), type=ACCESSOR_TYPE_VEC3))

        uv_view = add_view(texcoords.tobytes(), ARRAY_BUFFER)
        uv_accessor_index = len(accessors)
        accessors.append(Accessor(bufferView=uv_view, componentType=FLOAT, count=int(texcoords.shape[0]), type=ACCESSOR_TYPE_VEC2))

        uv1_accessor_index = None
        if texcoords1 is not None:
            uv1_view = add_view(texcoords1.tobytes(), ARRAY_BUFFER)
            uv1_accessor_index = len(accessors)
            accessors.append(
                Accessor(
                    bufferView=uv1_view,
                    componentType=FLOAT,
                    count=int(texcoords1.shape[0]),
                    type=ACCESSOR_TYPE_VEC2,
                )
            )

        idx_view = add_view(indices.tobytes(), ELEMENT_ARRAY_BUFFER)
        idx_accessor_index = len(accessors)
        accessors.append(Accessor(bufferView=idx_view, componentType=UNSIGNED_INT, count=int(indices.shape[0]), type=ACCESSOR_TYPE_SCALAR))

        attrs = Attributes(POSITION=pos_accessor_index, NORMAL=nrm_accessor_index, TEXCOORD_0=uv_accessor_index)
        if uv1_accessor_index is not None:
            attrs.TEXCOORD_1 = uv1_accessor_index
        prim = Primitive(attributes=attrs, indices=idx_accessor_index, material=0)
        gltf_meshes.append(Mesh(primitives=[prim], name=name))

    # Build nodes (hierarchy)
    for i, nd in enumerate(nodes):
        n = Node(name=nd.get("name") or f"node_{i}")
        if "translation" in nd and nd["translation"] is not None:
            t = nd["translation"]
            n.translation = [float(t[0]), float(t[1]), float(t[2])]
        if "rotation" in nd and nd["rotation"] is not None:
            r = nd["rotation"]
            n.rotation = [float(r[0]), float(r[1]), float(r[2]), float(r[3])]
        if "children" in nd and nd["children"]:
            n.children = [int(c) for c in nd["children"]]
        if "mesh" in nd and nd["mesh"] is not None:
            n.mesh = int(nd["mesh"])
        gltf_nodes.append(n)

    gltf = GLTF2(
        asset=Asset(version="2.0"),
        buffers=[Buffer(byteLength=0)],
        bufferViews=buffer_views,
        accessors=accessors,
        meshes=gltf_meshes,
        images=[
            Image(bufferView=img_view, mimeType="image/png", name=(f"{name_prefix}_tex" if name_prefix else None))
            if texture_png is not None
            else Image(uri=texture_path, name=(f"{name_prefix}_tex" if name_prefix else None))
        ],
        samplers=[Sampler(magFilter=9728, minFilter=9728, wrapS=33071, wrapT=33071)],
        textures=[Texture(sampler=0, source=0, name=(f"{name_prefix}_texture" if name_prefix else None))],
        materials=[
            Material(
                name=(f"{name_prefix}_mat" if name_prefix else None),
                pbrMetallicRoughness=PbrMetallicRoughness(
                    baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                    baseColorTexture=TextureInfo(index=0),
                    metallicFactor=0.0,
                    roughnessFactor=1.0,
                ),
                alphaMode=(str(alpha_mode) if alpha_mode else None),
                alphaCutoff=(float(alpha_cutoff) if alpha_cutoff is not None else None),
                doubleSided=True,
            )
        ],
        nodes=gltf_nodes,
        scenes=[Scene(nodes=[int(r) for r in root_node_ids])],
        scene=0,
    )

    gltf.buffers[0].byteLength = len(blob)
    gltf.set_binary_blob(bytes(blob))
    gltf.save_binary(str(output_glb))


def write_scene_gltf(
    output_gltf: str | Path,
    *,
    meshes: Iterable[dict[str, np.ndarray]],
    nodes: list[dict],
    root_node_ids: list[int],
    texture_png: bytes | None,
    texture_path: str | None,
    name_prefix: str | None = None,
    alpha_mode: str | None = None,
    alpha_cutoff: float | None = None,
) -> None:
    from pygltflib import (
        GLTF2,
        Accessor,
        Asset,
        Attributes,
        Buffer,
        BufferView,
        Image,
        Material,
        Mesh,
        Node,
        PbrMetallicRoughness,
        Primitive,
        Sampler,
        Scene,
        Texture,
        TextureInfo,
    )
    import math

    if (texture_png is None) == (texture_path is None):
        raise ValueError("Provide exactly one of texture_png or texture_path")

    output_gltf = Path(output_gltf)
    out_dir = output_gltf.parent
    bin_name = f"{output_gltf.stem}.bin"
    bin_path = out_dir / bin_name

    if texture_png is not None:
        tex_name = f"{output_gltf.stem}.png"
        (out_dir / tex_name).write_bytes(texture_png)
        texture_path = tex_name

    ARRAY_BUFFER = 34962
    ELEMENT_ARRAY_BUFFER = 34963
    UNSIGNED_INT = 5125
    FLOAT = 5126
    ACCESSOR_TYPE_SCALAR = "SCALAR"
    ACCESSOR_TYPE_VEC3 = "VEC3"
    ACCESSOR_TYPE_VEC2 = "VEC2"

    def _align4(n: int) -> int:
        return int(math.ceil(n / 4.0) * 4)

    blob = bytearray()
    buffer_views: list[BufferView] = []
    accessors: list[Accessor] = []
    gltf_meshes: list[Mesh] = []
    gltf_nodes: list[Node] = []

    def add_view(data: bytes, target: int | None) -> int:
        offset = len(blob)
        blob.extend(data)
        padded = _align4(len(blob))
        if padded != len(blob):
            blob.extend(b"\x00" * (padded - len(blob)))
        view_index = len(buffer_views)
        buffer_views.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(data), target=target))
        return view_index

    meshes_list = list(meshes)
    for m in meshes_list:
        name = m.get("name")
        positions = np.asarray(m["positions"], dtype=np.float32)
        normals = np.asarray(m["normals"], dtype=np.float32)
        texcoords = np.asarray(m["texcoords"], dtype=np.float32)
        texcoords1 = m.get("texcoords1")
        if texcoords1 is not None:
            texcoords1 = np.asarray(texcoords1, dtype=np.float32)
        indices = np.asarray(m["indices"], dtype=np.uint32)

        pos_view = add_view(positions.tobytes(), ARRAY_BUFFER)
        pos_accessor_index = len(accessors)
        accessors.append(
            Accessor(
                bufferView=pos_view,
                componentType=FLOAT,
                count=int(positions.shape[0]),
                type=ACCESSOR_TYPE_VEC3,
                min=positions.min(axis=0).tolist(),
                max=positions.max(axis=0).tolist(),
            )
        )

        nrm_view = add_view(normals.tobytes(), ARRAY_BUFFER)
        nrm_accessor_index = len(accessors)
        accessors.append(Accessor(bufferView=nrm_view, componentType=FLOAT, count=int(normals.shape[0]), type=ACCESSOR_TYPE_VEC3))

        uv_view = add_view(texcoords.tobytes(), ARRAY_BUFFER)
        uv_accessor_index = len(accessors)
        accessors.append(Accessor(bufferView=uv_view, componentType=FLOAT, count=int(texcoords.shape[0]), type=ACCESSOR_TYPE_VEC2))

        uv1_accessor_index = None
        if texcoords1 is not None:
            uv1_view = add_view(texcoords1.tobytes(), ARRAY_BUFFER)
            uv1_accessor_index = len(accessors)
            accessors.append(Accessor(bufferView=uv1_view, componentType=FLOAT, count=int(texcoords1.shape[0]), type=ACCESSOR_TYPE_VEC2))

        idx_view = add_view(indices.tobytes(), ELEMENT_ARRAY_BUFFER)
        idx_accessor_index = len(accessors)
        accessors.append(Accessor(bufferView=idx_view, componentType=UNSIGNED_INT, count=int(indices.shape[0]), type=ACCESSOR_TYPE_SCALAR))

        attrs = Attributes(POSITION=pos_accessor_index, NORMAL=nrm_accessor_index, TEXCOORD_0=uv_accessor_index)
        if uv1_accessor_index is not None:
            attrs.TEXCOORD_1 = uv1_accessor_index
        prim = Primitive(attributes=attrs, indices=idx_accessor_index, material=0)
        gltf_meshes.append(Mesh(primitives=[prim], name=name))

    # Build nodes (hierarchy)
    for i, nd in enumerate(nodes):
        n = Node(name=nd.get("name") or f"node_{i}")
        if "translation" in nd and nd["translation"] is not None:
            t = nd["translation"]
            n.translation = [float(t[0]), float(t[1]), float(t[2])]
        if "rotation" in nd and nd["rotation"] is not None:
            r = nd["rotation"]
            n.rotation = [float(r[0]), float(r[1]), float(r[2]), float(r[3])]
        if "children" in nd and nd["children"]:
            n.children = [int(c) for c in nd["children"]]
        if "mesh" in nd and nd["mesh"] is not None:
            n.mesh = int(nd["mesh"])
        gltf_nodes.append(n)

    gltf = GLTF2(
        asset=Asset(version="2.0"),
        buffers=[Buffer(byteLength=0, uri=bin_name)],
        bufferViews=buffer_views,
        accessors=accessors,
        meshes=gltf_meshes,
        images=[Image(uri=str(texture_path), name=(f"{name_prefix}_tex" if name_prefix else None))],
        samplers=[Sampler(magFilter=9728, minFilter=9728, wrapS=33071, wrapT=33071)],
        textures=[Texture(sampler=0, source=0, name=(f"{name_prefix}_texture" if name_prefix else None))],
        materials=[
            Material(
                name=(f"{name_prefix}_mat" if name_prefix else None),
                pbrMetallicRoughness=PbrMetallicRoughness(
                    baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                    baseColorTexture=TextureInfo(index=0),
                    metallicFactor=0.0,
                    roughnessFactor=1.0,
                ),
                alphaMode=(str(alpha_mode) if alpha_mode else None),
                alphaCutoff=(float(alpha_cutoff) if alpha_cutoff is not None else None),
                doubleSided=True,
            )
        ],
        nodes=gltf_nodes,
        scenes=[Scene(nodes=[int(r) for r in root_node_ids])],
        scene=0,
    )

    gltf.buffers[0].byteLength = len(blob)
    bin_path.write_bytes(bytes(blob))
    gltf.save(str(output_gltf))


def write_meshes(
    output_glb: str | Path,
    *,
    meshes: Iterable[dict[str, np.ndarray]],
    texture_png: bytes | None,
    texture_path: str | None,
    name_prefix: str | None = None,
) -> None:
    """Write meshes to a GLB, embedding or referencing the texture."""

    write_glb_scene(
        str(output_glb),
        meshes=list(meshes),
        texture_png=texture_png,
        texture_uri=texture_path,
        name_prefix=name_prefix,
    )
