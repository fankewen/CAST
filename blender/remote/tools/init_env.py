"""Generate local secrets; never prints credentials or overwrites an existing file."""
import os
from pathlib import Path
import secrets

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    target=ROOT/'.env'
    text=(ROOT/'.env.example').read_text()
    text=text.replace('CAST_API_KEY=\n','CAST_API_KEY='+secrets.token_hex(32)+'\n')
    text=text.replace('CAST_ADMIN_PASSWORD=\n','CAST_ADMIN_PASSWORD='+secrets.token_urlsafe(32)+'\n')
    fd=os.open(target,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w') as f:f.write(text)
    print('Created .env (0600). Read it locally to obtain your API/admin credentials.')
