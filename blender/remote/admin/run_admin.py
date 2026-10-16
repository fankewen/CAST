import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from settings import SETTINGS
import uvicorn

if __name__=='__main__':
    uvicorn.run('admin.app:app',host=os.environ.get('CAST_ADMIN_HOST','127.0.0.1'),
                port=SETTINGS.admin_port,workers=1,access_log=False,timeout_keep_alive=5,
                limit_concurrency=64,proxy_headers=False)
