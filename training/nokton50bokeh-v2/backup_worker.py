"""Snapshot stable checkpoint files AND optimizer-state folders to mounted Drive."""
from pathlib import Path
import os,time,shutil,json
r=Path('/content/nokton50bokeh-v2');backup=Path(os.environ['BACKUP_DIR']);previous={}
while not (r/'STOP_BACKUP').exists():
 for p in (r/'output').glob('*'):
  if not (p.is_dir() or p.suffix=='.safetensors'):continue
  files=sorted(x for x in p.rglob('*') if x.is_file()) if p.is_dir() else [p]
  signature=[[str(x.relative_to(p)) if p.is_dir() else p.name,x.stat().st_size,x.stat().st_mtime_ns] for x in files]
  if not files or previous.get(p.name)!=signature:previous[p.name]=signature;continue
  dest=backup/p.name;marker=backup/(p.name+'.backup.json')
  if marker.exists() and json.loads(marker.read_text())==signature:continue
  temp=backup/(p.name+'.partial')
  if temp.is_dir():shutil.rmtree(temp)
  elif temp.exists():temp.unlink()
  if p.is_dir():shutil.copytree(p,temp)
  else:shutil.copy2(p,temp)
  if dest.is_dir():shutil.rmtree(dest)
  temp.replace(dest);marker.write_text(json.dumps(signature));print('Backed up',p.name,flush=True)
 time.sleep(30)
