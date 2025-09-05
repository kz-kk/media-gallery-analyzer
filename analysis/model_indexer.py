#!/usr/bin/env python3
# Lightweight GLB analyzer without external deps

import sys
import os
import json
import struct

def read_glb_json(path):
    with open(path, 'rb') as f:
        header = f.read(12)
        if len(header) < 12:
            raise ValueError('File too small')
        magic, version, length = struct.unpack('<III', header)
        if magic != 0x46546C67:  # 'glTF'
            raise ValueError('Not a GLB file')
        if version != 2:
            raise ValueError(f'Unsupported GLB version: {version}')

        json_chunk = None
        # Iterate chunks
        offset = 12
        while offset < length:
            f.seek(offset)
            chunk_header = f.read(8)
            if len(chunk_header) < 8:
                break
            chunk_len, chunk_type = struct.unpack('<II', chunk_header)
            data = f.read(chunk_len)
            if chunk_type == 0x4E4F534A:  # JSON
                try:
                    json_chunk = json.loads(data.decode('utf-8'))
                except Exception as e:
                    raise ValueError(f'Failed to parse GLB JSON: {e}')
                break
            offset += 8 + chunk_len
    if json_chunk is None:
        raise ValueError('JSON chunk not found')
    return json_chunk

def analyze_gltf(g):
    stats = {}
    meshes = g.get('meshes', [])
    materials = g.get('materials', [])
    nodes = g.get('nodes', [])
    scenes = g.get('scenes', [])
    animations = g.get('animations', [])
    accessors = g.get('accessors', [])

    stats['mesh_count'] = len(meshes)
    stats['material_count'] = len(materials)
    stats['node_count'] = len(nodes)
    stats['scene_count'] = len(scenes)
    stats['animation_count'] = len(animations)
    stats['extensions_used'] = g.get('extensionsUsed', [])

    # Names summary
    stats['mesh_names'] = [m.get('name') for m in meshes if m.get('name')]
    stats['material_names'] = [m.get('name') for m in materials if m.get('name')]

    # Vertices / indices estimate
    total_vertices = 0
    total_triangles = 0
    bbox_min = None
    bbox_max = None

    # Build accessor cache
    def get_accessor(idx):
        if idx is None: return None
        if 0 <= idx < len(accessors):
            return accessors[idx]
        return None

    # Iterate primitives
    for m in meshes:
        for prim in m.get('primitives', []):
            attrs = prim.get('attributes', {}) or {}
            pos_idx = attrs.get('POSITION')
            acc = get_accessor(pos_idx)
            if acc and isinstance(acc.get('count'), int):
                total_vertices += int(acc['count'])
                # Bounds
                if isinstance(acc.get('min'), list) and isinstance(acc.get('max'), list):
                    amin = acc['min']
                    amax = acc['max']
                    if bbox_min is None:
                        bbox_min = list(amin)
                        bbox_max = list(amax)
                    else:
                        for i in range(min(len(bbox_min), len(amin))):
                            bbox_min[i] = min(bbox_min[i], amin[i])
                            bbox_max[i] = max(bbox_max[i], amax[i])

            # Triangle estimate from indices if present
            idx_acc = get_accessor(prim.get('indices'))
            if idx_acc and isinstance(idx_acc.get('count'), int):
                total_triangles += int(idx_acc['count']) // 3

            # If KHR_draco_mesh_compression is used, attributes may be under extensions
            ext = (prim.get('extensions') or {}).get('KHR_draco_mesh_compression')
            if ext:
                # We cannot decode Draco here; try to use provided attributes mapping counts if available
                # Fallback: mark as using_draco
                stats['uses_draco'] = True

    stats['total_vertices'] = total_vertices
    stats['total_triangles_estimate'] = total_triangles
    if bbox_min is not None and bbox_max is not None:
        stats['bbox_min'] = bbox_min
        stats['bbox_max'] = bbox_max

    # Images / textures summary
    stats['image_count'] = len(g.get('images', []) or [])
    stats['texture_count'] = len(g.get('textures', []) or [])

    return stats

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='Path to .glb file')
    args = p.parse_args()

    path = args.model
    result = {
        'success': False,
    }
    try:
        print('MODEL_PROGRESS: start', flush=True)
        print('MODEL_PROGRESS: reading', flush=True)
        g = read_glb_json(path)
        print('MODEL_PROGRESS: parsing', flush=True)
        stats = analyze_gltf(g)
        print('MODEL_PROGRESS: analyzing', flush=True)
        file_size = os.path.getsize(path)
        stats['file_size_bytes'] = file_size

        # Build caption and tags
        caption = f"3D model with {stats['mesh_count']} meshes, {stats['material_count']} materials"
        if stats.get('animation_count'):
            caption += f", {stats['animation_count']} animations"
        tags = ['3D', 'glb']
        if stats.get('uses_draco'): tags.append('draco')
        tags.append(f"meshes:{stats['mesh_count']}")
        tags.append(f"materials:{stats['material_count']}")
        if stats.get('total_vertices'):
            tags.append(f"vertices:{stats['total_vertices']}")
        if stats.get('total_triangles_estimate'):
            tags.append(f"triangles~:{stats['total_triangles_estimate']}")

        result.update({
            'success': True,
            'caption': caption,
            'tags': tags,
            'model': 'glb-parser',
            'stats': stats,
        })
        print('MODEL_PROGRESS: done', flush=True)
    except Exception as e:
        result = {'success': False, 'error': str(e)}

    print('MODEL_ANALYSIS_RESULT: ' + json.dumps(result, ensure_ascii=False))

if __name__ == '__main__':
    main()
