"""Stdlib-only bounded HTTP pipeline for a Blender background worker.

Use one instance per camera. Call submit in capture order; keep the returned
futures in a deque and APPLY results in that same order. No bpy in HTTP threads.
"""
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import threading
import time
from urllib.parse import urlsplit


class MotionHTTPError(RuntimeError):
    def __init__(self, status, detail):
        self.status, self.detail = status, detail
        super().__init__(f"HTTP {status}: {detail}")


class PipelinedMotionClient:
    def __init__(self,url,api_key,window=2,timeout=900):
        if not 1<=window<=10:
            raise ValueError('window must be 1..10; use 2 for low RTT or 6..10 for high RTT')
        p=urlsplit(url)
        if p.scheme not in ('http','https'):
            raise ValueError('use http:// or https://')
        self.conn_class=http.client.HTTPSConnection if p.scheme=='https' else http.client.HTTPConnection
        self.host,self.port,self.base=p.hostname,p.port,p.path.rstrip('/')
        self.api_key,self.timeout=api_key,timeout
        self.pool=ThreadPoolExecutor(max_workers=window,thread_name_prefix='cast-http')
        self.local=threading.local()
        self.connections=[]
        self.window=window
        self.lock=threading.Lock()
        self.pending=0
        self.failed=False
        self.closed=False
        self.sid=self.epoch=None
        self.seq=0

    def _request(self,method,path,body=None,content_type='application/json'):
        # One persistent connection per worker, never shared concurrently.
        # Uvicorn's keep-alive timeout is 5s. Reopen idle connections BEFORE
        # sending, rather than blindly retrying a potentially consumed POST.
        if hasattr(self.local,'conn') and time.monotonic()-getattr(self.local,'last_used',0)>=4:
            self.local.conn.close()
            self.connections.remove(self.local.conn)
            del self.local.conn
        if not hasattr(self.local,'conn'):
            self.local.conn=self.conn_class(self.host,self.port,timeout=self.timeout)
            self.connections.append(self.local.conn)
        conn=self.local.conn
        try:
            conn.request(method,self.base+path,body=body,headers={
                'X-API-Key':self.api_key,'Content-Type':content_type})
            r=conn.getresponse(); raw=r.read()
            self.local.last_used=time.monotonic()
            try:
                data=json.loads(raw)
            except ValueError:
                raise MotionHTTPError(r.status,'non-JSON server response')
            if r.status>=400:
                raise MotionHTTPError(r.status,data.get('detail',data))
            return data
        except Exception:
            conn.close();self.connections.remove(conn);del self.local.conn
            raise

    def start(self,rig,target_mode='human'):
        if self.sid or self.closed:
            raise RuntimeError('close old client or create a fresh client')
        if not isinstance(rig,dict):
            raise ValueError('rig JSON is required; export the selected Blender armature first')
        if target_mode not in ('human','generic'):
            raise ValueError('target_mode must be human or generic')
        data=self._request('POST','/v1/sessions',json.dumps({'target_mode':target_mode,'rig':rig}).encode())
        self.sid,self.epoch=data['session_id'],data.get('epoch')
        if not self.epoch or data.get('pipeline_capacity',1)<self.window:
            self._request('DELETE',f'/v1/sessions/{self.sid}');self.sid=None
            raise RuntimeError('server does not advertise compatible pipeline support')
        return data

    def adopt_session(self, session_id, epoch, capacity=None):
        """Attach to an idle initialized session and resume its actual sequence."""
        if self.sid or self.closed:
            raise RuntimeError('client already bound or closed')
        self.sid = session_id
        try:
            status = self.status()
            pipeline = status['pipeline']
            if pipeline['epoch'] != epoch or pipeline['pending'] or pipeline.get('failed'):
                raise RuntimeError('session is stale, busy or failed; initialize again')
            if pipeline['capacity'] < self.window:
                raise RuntimeError('server pipeline capacity is smaller than client window')
            self.epoch, self.seq = epoch, int(pipeline['next_seq'])
            return status
        except BaseException:
            self.sid = None
            raise

    def submit(self,jpeg_bytes,timestamp_ms=None):
        with self.lock:
            if self.failed or not self.sid or self.closed:
                raise RuntimeError('not ready or previous request failed; drain then reset')
            if self.pending>=self.window:
                raise BufferError('client window full; keep latest unsubmitted image, do not increment seq')
            seq=self.seq;self.seq+=1;self.pending+=1
        epoch = self.epoch
        path=f'/v1/sessions/{self.sid}/frames?seq={seq}&epoch={epoch}'
        if timestamp_ms is not None:
            path+=f'&timestamp_ms={float(timestamp_ms)}'
        def send():
            try:
                result=self._request('POST',path,jpeg_bytes,'image/jpeg')
                if result.get('seq')!=seq or result.get('epoch')!=epoch:
                    raise RuntimeError('server response sequence or epoch mismatch')
                return result
            except Exception:
                with self.lock:
                    self.failed=True
                raise
            finally:
                # Release capacity before Future.result() wakes its consumer.
                with self.lock:
                    self.pending-=1
        future=self.pool.submit(send)
        def done(f):
            if f.cancelled():
                with self.lock:
                    self.pending-=1
                    self.failed=True
        future.add_done_callback(done)
        return future

    def reset(self):
        with self.lock:
            if self.pending:
                raise RuntimeError('drain all returned futures before reset')
        # On ambiguous network timeout, the server may still have jobs; retry
        # reset only after status shows pending=0, not by replaying old frames.
        data=self._request('POST',f'/v1/sessions/{self.sid}/reset')
        self.epoch=data['epoch'];self.seq=0;self.failed=False
        return data

    def status(self):
        return self._request('GET',f'/v1/sessions/{self.sid}/status')

    def close(self, delete_session=True):
        self.closed=True
        self.pool.shutdown(wait=True)
        try:
            if self.sid and delete_session:
                self._request('DELETE',f'/v1/sessions/{self.sid}')
                self.sid=None
        finally:
            for conn in self.connections:
                conn.close()
