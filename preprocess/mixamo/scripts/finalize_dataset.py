"""Complete rendered per-character caches and build directed cross-skeleton pairs."""
from pathlib import Path
import argparse, io, json, tarfile
import numpy as np
from PIL import Image

def main(root):
    names=sorted(p.parent.name for p in root.glob('*/meta.json'))
    metas={n:json.loads((root/n/'meta.json').read_text()) for n in names}
    expected={m['clip_name']:(m['frames'],m['fps']) for m in metas[names[0]]['motions']}
    for n,meta in metas.items():
        assert {m['clip_name']:(m['frames'],m['fps']) for m in meta['motions']}==expected, n
    validation=[]
    for name,meta in metas.items():
        base=root/name
        (base/'images').mkdir(exist_ok=True)
        with tarfile.open(base/'images'/'oblique15_el18.tar','w') as tf:
            for m in meta['motions']:
                clip=m['clip_name']; boxes=[]
                for folder in ['rgb','mask']:
                    (base/folder/clip).mkdir(parents=True,exist_ok=True)
                assert len(list((base/'rgba'/clip).glob('*.png')))==m['frames']
                for i in range(m['frames']):
                    fn=f'{i:06d}.png'
                    with Image.open(base/'rgba'/clip/fn) as src:
                        im=src.convert('RGBA')
                    assert im.size==(512,512)
                    alpha=im.getchannel('A'); box=alpha.getbbox()
                    if box is None: raise ValueError(f'Empty silhouette: {name}/{clip}/{i}')
                    boxes.append(box)
                    rgb=Image.new('RGB',im.size,'white'); rgb.paste(im.convert('RGB'),mask=alpha)
                    mask=Image.fromarray(np.uint8(np.asarray(alpha)>0)*255)
                    rgb.save(base/'rgb'/clip/fn); mask.save(base/'mask'/clip/fn)
                    for ext,image,fmt in [('.jpg',rgb,'JPEG'),('.png',mask,'PNG')]:
                        b=io.BytesIO(); image.save(b,format=fmt,**({'quality':95,'subsampling':0} if fmt=='JPEG' else {}))
                        data=b.getvalue(); info=tarfile.TarInfo(f'{clip}.{i:06d}{ext}'); info.size=len(data); tf.addfile(info,io.BytesIO(data))
                m['camera']=m['camera_folder']='oblique15_el18'; m['image']['tar']='images/oblique15_el18.tar'
                validation.append(dict(character=name,clip=clip,frames=m['frames'],fps=m['fps'],edge_touch_frames=sum(x<=0 or y<=0 or r>=512 or b>=512 for x,y,r,b in boxes)))
        meta['cache_status']='COMPLETE'
        (base/'meta.json').write_text(json.dumps(meta,indent=2))
        print('CACHE_COMPLETE',name,flush=True)
    pairs=[]
    for source in names:
        for target in names:
            if source==target: continue
            for clip,(frames,fps) in expected.items():
                pairs.append(dict(pair_id=f'{source}_to_{target}__{clip}',source=source,target=target,clip=clip,frames=frames,fps=fps,source_rgb=f'{source}/rgb/{clip}',source_mask=f'{source}/mask/{clip}',target_static=f'{target}/static.npy',target_reference=f'{target}/reference/{clip}.npz'))
    assert len(pairs)==300
    (root/'pairs.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in pairs))
    (root/'validation.json').write_text(json.dumps(dict(status='COMPLETE',characters=names,clips=validation,pair_count=len(pairs),frames=sum(r['frames'] for r in validation)),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--cache',type=Path,required=True)
    main(p.parse_args().cache)
