"""Weld only the elbows, wrists and knees of the twelve-part hero.

Preserve shoulder/hip attachments and all head/body geometry. UVs stay on loops;
the joint cross-sections share actual vertices with two-bone skin weights.
"""
import bmesh


def sew_limb_joints(mesh):
    bm = bmesh.new()
    bm.from_mesh(mesh.data)
    deform = bm.verts.layers.deform.verify()
    report = []
    sewn_faces = []
    for side in ('L', 'R'):
        for kind, names, joints in (
            ('arm', ['upper_arm.'+side, 'forearm.'+side, 'hand.'+side], [(.66,.018),(.52,.012)]),
            ('leg', ['thigh.'+side, 'shin.'+side], [(.14,.018)]),
        ):
            ids = [mesh.vertex_groups[n].index for n in names]

            def limb_verts():
                return [v for v in bm.verts if any(v[deform].get(i,0)>0 for i in ids)]

            verts = set(limb_verts())
            faces = [f for f in bm.faces if all(v in verts for v in f.verts)]
            caps = [f for f in faces if any(all(abs(v.co.z-height)<1e-5 for v in f.verts)
                                          for height,width in joints)]
            assert len(caps)>=2*len(joints),(names,'missing internal caps',len(caps))
            bmesh.ops.delete(bm,geom=caps,context='FACES_ONLY')
            loose=[e for e in bm.edges if not e.link_faces and all(v in verts for v in e.verts)]
            if loose:bmesh.ops.delete(bm,geom=loose,context='EDGES')
            bmesh.ops.remove_doubles(bm,verts=limb_verts(),dist=1e-5)
            # Support loops keep the blend confined to the joint, avoiding rubbery limbs.
            for height,width in joints:
                for z in (height-width,height+width):
                    vs=set(limb_verts())
                    edges=[e for e in bm.edges if all(v in vs for v in e.verts)]
                    faces=[f for f in bm.faces if all(v in vs for v in f.verts)]
                    bmesh.ops.bisect_plane(bm,geom=list(vs)+edges+faces,dist=1e-6,
                        plane_co=(0,0,z),plane_no=(0,0,1),clear_inner=False,clear_outer=False)
            vs = limb_verts()
            for v in vs:
                weights = v[deform]
                weights.clear()
                assigned=False
                for joint_index,(height,width) in enumerate(joints):
                    if abs(v.co.z-height) <= width+1e-5:
                        t=max(0,min(1,(v.co.z-height+width)/(2*width)))
                        t=t*t*(3-2*t)
                        if t>1e-7:weights[ids[joint_index]]=t
                        if 1-t>1e-7:weights[ids[joint_index+1]]=1-t
                        assigned=True
                        break
                if not assigned:
                    index=sum(v.co.z<height for height,width in joints)
                    weights[ids[index]]=1
            vs=set(vs)
            edges=[e for e in bm.edges if all(v in vs for v in e.verts)]
            assert all(e.is_manifold for e in edges),(names,'open seam')
            # A single connected, closed limb; body attachments were never included.
            todo=[next(iter(vs))]
            seen=set(todo)
            while todo:
                v=todo.pop()
                for e in v.link_edges:
                    neighbor=e.other_vert(v)
                    if neighbor in vs and neighbor not in seen:
                        seen.add(neighbor);todo.append(neighbor)
            assert seen==vs,(names,'disconnected limb')
            for height,width in joints:
                ring=[v for v in vs if abs(v.co.z-height)<1e-5]
                assert len(ring)==4,(names,height,'expected a single welded ring',len(ring))
                assert all(len(v[deform])==2 for v in ring),(names,'missing blended weights')
            report.append({'limb':kind+'.'+side,'vertices':len(vs),'joints':len(joints),
                           'closed':True,'connected':True})
            sewn_faces.extend(f for f in bm.faces if all(v in vs for v in f.verts))
    # Imported head/body faces have split vertices. Reorienting disconnected
    # islands would flip some of their existing outward normals.
    bmesh.ops.recalc_face_normals(bm,faces=sewn_faces)
    bm.to_mesh(mesh.data)
    bm.free()
    mesh.data.update()
    return report
