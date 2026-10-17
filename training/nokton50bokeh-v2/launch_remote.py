"""Run only after upload/setup and explicit launch request; mounts must already exist."""
from pathlib import Path
import subprocess,os
r=Path('/content/nokton50bokeh-v2');backup=Path('/content/drive/MyDrive/nokton50bokeh-v2')
assert Path('/content/drive/MyDrive').is_dir(),'Mount Google Drive first'
assert not (r/'TRAINING_STARTED').exists(),'Already started; inspect status before resuming'
backup.mkdir(exist_ok=True);env=os.environ.copy();env.update(BACKUP_DIR=str(backup),PYTHONUNBUFFERED='1')
subprocess.Popen(['python',str(r/'backup_worker.py')],env=env,stdout=open(r/'backup.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
p=subprocess.Popen(['bash',str(r/'train.sh')],env=env,stdout=open(r/'train.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
(r/'training.pid').write_text(str(p.pid));print('Training PID',p.pid)
