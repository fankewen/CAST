"""No weights required; temporary state and ephemeral loopback sockets only."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
if __name__=='__main__':
    with tempfile.TemporaryDirectory(prefix='cast-tests-') as temp:
        os.environ['CAST_ENV_FILE']=str(Path(temp)/'absent.env')
        os.environ['CAST_STATE_DIR']=temp
        os.environ['CAST_ADMIN_STATE_DIR']=str(Path(temp)/'admin-local')
        os.environ['CAST_ASSETS_DIR']=str(Path(temp)/'assets')
        import settings
        suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
