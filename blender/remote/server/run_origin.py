"""Start pinned Uvicorn with separated TCP/request protection and origin audit."""
import os
import uvicorn
from settings import SETTINGS
from origin_http import OriginH11Protocol, configure_audit


def main():
    if uvicorn.__version__ != "0.34.2":
        raise RuntimeError("origin_http is validated against uvicorn==0.34.2; retest before upgrade")
    home = SETTINGS.state_dir
    sink = configure_audit(home / "logs/http_origin.jsonl")
    try:
        uvicorn.run("service:app", host=os.environ.get("CAST_HOST", "127.0.0.1"),
                    port=int(os.environ.get("CAST_PORT", "8080")), workers=1,
                    http=OriginH11Protocol, ws="none", access_log=False,
                    proxy_headers=False, limit_concurrency=None, timeout_keep_alive=5)
    finally:
        sink.close()


if __name__ == "__main__":
    main()
