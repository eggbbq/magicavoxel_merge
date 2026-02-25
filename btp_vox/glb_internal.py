from __future__ import annotations

import math

import numpy as np
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


ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963

UNSIGNED_INT = 5125
FLOAT = 5126

ACCESSOR_TYPE_SCALAR = "SCALAR"
ACCESSOR_TYPE_VEC3 = "VEC3"
ACCESSOR_TYPE_VEC4 = "VEC4"
ACCESSOR_TYPE_VEC2 = "VEC2"


def _align4(n: int) -> int:
    return int(math.ceil(n / 4.0) * 4)


def write_glb_scene(
    output_path: str,
    *,
    meshes: list[dict[str, np.ndarray]],
    texture_png: bytes | None = None,
    texture_uri: str | None = None,
    name_prefix: str | None = None,
) -> None:
    if (texture_png is None) == (texture_uri is None):
        raise ValueError("Provide exactly one of texture_png or texture_uri")

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

    img_view: int | None = None
    if texture_png is not None:
        img_view = add_view(texture_png, None)

    for m in meshes:
        name = m.get("name")
        translation = m.get("translation")
        rotation = m.get("rotation")
        positions = np.asarray(m["positions"], dtype=np.float32)
        normals = np.asarray(m["normals"], dtype=np.float32)
        texcoords = np.asarray(m["texcoords"], dtype=np.float32)
        indices = np.asarray(m["indices"], dtype=np.uint32)
        color0 = m.get("color0")
        tangent = m.get("tangent")

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must be (N,3)")
        if normals.shape != positions.shape:
            raise ValueError("normals must match positions")
        if texcoords.ndim != 2 or texcoords.shape[1] != 2:
            raise ValueError("texcoords must be (N,2)")
        if indices.ndim != 1:
            raise ValueError("indices must be (M,)")

        if color0 is not None:
            color0 = np.asarray(color0, dtype=np.float32)
            if color0.shape != (positions.shape[0], 4):
                raise ValueError("color0 must be (N,4) to match positions")
        if tangent is not None:
            tangent = np.asarray(tangent, dtype=np.float32)
            if tangent.shape != (positions.shape[0], 4):
                raise ValueError("tangent must be (N,4) to match positions")

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

        col_accessor_index = None
        if color0 is not None:
            col_view = add_view(color0.tobytes(), ARRAY_BUFFER)
            col_accessor_index = len(accessors)
            accessors.append(Accessor(bufferView=col_view, componentType=FLOAT, count=int(color0.shape[0]), type=ACCESSOR_TYPE_VEC4))

        tan_accessor_index = None
        if tangent is not None:
            tan_view = add_view(tangent.tobytes(), ARRAY_BUFFER)
            tan_accessor_index = len(accessors)
            accessors.append(Accessor(bufferView=tan_view, componentType=FLOAT, count=int(tangent.shape[0]), type=ACCESSOR_TYPE_VEC4))

        idx_view = add_view(indices.tobytes(), ELEMENT_ARRAY_BUFFER)
        idx_accessor_index = len(accessors)
        accessors.append(Accessor(bufferView=idx_view, componentType=UNSIGNED_INT, count=int(indices.shape[0]), type=ACCESSOR_TYPE_SCALAR))

        attrs = Attributes(POSITION=pos_accessor_index, NORMAL=nrm_accessor_index, TEXCOORD_0=uv_accessor_index)
        if col_accessor_index is not None:
            attrs.COLOR_0 = col_accessor_index
        if tan_accessor_index is not None:
            attrs.TANGENT = tan_accessor_index

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
        buffers=[Buffer(byteLength=0)],
        bufferViews=buffer_views,
        accessors=accessors,
        meshes=gltf_meshes,
        images=[
            Image(bufferView=img_view, mimeType="image/png", name=(f"{name_prefix}_tex" if name_prefix else None))
            if texture_png is not None
            else Image(uri=texture_uri, name=(f"{name_prefix}_tex" if name_prefix else None))
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
                doubleSided=True,
            )
        ],
        nodes=nodes,
        scenes=[Scene(nodes=list(range(len(nodes))))],
        scene=0,
    )

    gltf.buffers[0].byteLength = len(blob)
    gltf.set_binary_blob(bytes(blob))
    gltf.save_binary(output_path)
