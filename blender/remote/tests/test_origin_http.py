"""Isolated real-socket tests of origin admission; no model, no production port."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
import uvicorn
from uvicorn.server import ServerState
import origin_http as origin


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.old_limits=(origin.ACTIVE_LIMIT,origin.TCP_LIMIT)
        self.old_header=origin.HEADER_TIMEOUT
        origin.ACTIVE_LIMIT,origin.TCP_LIMIT=4,100
        self.sink=origin.configure_audit(Path(self.temp.name)/'audit.jsonl')
        self.gate=asyncio.Event()
        self.started=0
        async def app(scope,receive,send):
            if scope['path'].endswith('/frames'):
                self.started+=1
                await self.gate.wait()
            body=b'ok'
            await send({'type':'http.response.start','status':200,'headers':[(b'content-length',b'2')]})
            await send({'type':'http.response.body','body':body})
        self.state=ServerState()
        cfg=uvicorn.Config(app,http=origin.OriginH11Protocol,ws='none',lifespan='off',limit_concurrency=None,access_log=False,timeout_keep_alive=30)
        cfg.load()
        self.server=await asyncio.get_running_loop().create_server(
            lambda:origin.OriginH11Protocol(cfg,self.state,{},_loop=asyncio.get_running_loop()),'127.0.0.1',0)
        self.port=self.server.sockets[0].getsockname()[1]
        self.open=[]
        self.path='/v1/sessions/'+'a'*32+'/frames'

    async def asyncTearDown(self):
        self.gate.set()
        for r,w in self.open:w.close()
        for r,w in self.open:
            try:await w.wait_closed()
            except OSError:pass
        await asyncio.sleep(.05)
        self.server.close();await self.server.wait_closed()
        if self.state.tasks:
            await asyncio.gather(*tuple(self.state.tasks),return_exceptions=True)
        self.sink.close();origin.AUDIT=None
        origin.ACTIVE_LIMIT,origin.TCP_LIMIT=self.old_limits
        origin.HEADER_TIMEOUT=self.old_header
        self.temp.cleanup()

    async def connect(self):
        pair=await asyncio.open_connection('127.0.0.1',self.port);self.open.append(pair);return pair

    async def send(self,pair,path='/healthz',extra=''):
        pair[1].write(f'GET {path} HTTP/1.1\r\nHost: localhost\r\n{extra}\r\n'.encode())
        await pair[1].drain()

    async def response(self,pair):
        header=await asyncio.wait_for(pair[0].readuntil(b'\r\n\r\n'),3)
        lines=header.decode().split('\r\n');code=int(lines[0].split()[1]);h={}
        for line in lines[1:]:
            if ':' in line:
                k,v=line.split(':',1);h[k.lower()]=v.strip()
        body=await asyncio.wait_for(pair[0].readexactly(int(h.get('content-length',0))),3)
        return code,h,body

    async def wait_started(self,n):
        async def wait():
            while self.started<n:await asyncio.sleep(.002)
        await asyncio.wait_for(wait(),3)

    async def test_more_than_64_idle_connections_do_not_limit_requests(self):
        for _ in range(75):
            pair=await self.connect();await self.send(pair);self.assertEqual((await self.response(pair))[0],200)
        pair=await self.connect();await self.send(pair,'/v1/sessions')
        code,h,_=await self.response(pair)
        self.assertEqual(code,200);self.assertEqual(h['x-cast-policy'],'active-request-v1')
        self.assertGreaterEqual(len(self.state.connections),76)
        self.assertEqual(origin.snapshot()['active_requests'],0)

    async def test_active_limit_has_request_correlation_and_health_reserve(self):
        held=[]
        for i in range(4):
            pair=await self.connect();await self.send(pair,self.path+f'?seq={i}');held.append(pair)
        await self.wait_started(4)
        reject=await self.connect()
        await self.send(reject,self.path+'?seq=427&epoch='+'b'*32,
                        'x-rio-seq: synthetic-test-rio\r\nCookie: DO_NOT_LOG_COOKIE\r\nX-API-Key: DO_NOT_LOG_KEY\r\n')
        code,h,body=await self.response(reject)
        self.assertEqual(code,503);self.assertEqual(body,b'Service Unavailable')
        self.assertEqual(h['x-cast-reject-reason'],'active-request-limit')
        self.assertEqual(h['x-cast-rio-seq'],'synthetic-test-rio')
        health=await self.connect();await self.send(health);self.assertEqual((await self.response(health))[0],200)
        self.gate.set()
        for pair in held:self.assertEqual((await self.response(pair))[0],200)
        await asyncio.to_thread(self.sink.queue.join)
        raw=(Path(self.temp.name)/'audit.jsonl').read_text()
        self.assertNotIn('DO_NOT_LOG',raw)
        rows=[json.loads(l) for l in raw.splitlines()]
        r=next(r for r in rows if r['event']=='request_rejected')
        self.assertEqual(r['seq'],'427');self.assertEqual(r['epoch'],'b'*32)
        self.assertEqual(r['rio_seq'],'synthetic-test-rio');self.assertEqual(r['active_requests'],4)
        self.assertGreaterEqual(r['uvicorn_tasks'],4)
        self.assertEqual(r['request_id'],h['x-cast-request-id'])

    async def test_tcp_pressure_evicts_idle_not_active(self):
        held=[]
        for _ in range(2):
            pair=await self.connect();await self.send(pair,self.path);held.append(pair)
        await self.wait_started(2)
        for _ in range(6):
            pair=await self.connect();await self.send(pair);await self.response(pair)
        self.state.cast_origin.tcp_limit=8
        pair=await self.connect();await self.send(pair);self.assertEqual((await self.response(pair))[0],200)
        self.assertEqual(self.state.cast_origin.idle_evicted,1)
        self.assertEqual(self.state.cast_origin.active,2)
        self.gate.set()
        for pair in held:self.assertEqual((await self.response(pair))[0],200)

    async def test_tcp_cap_without_idle_has_origin_marker(self):
        first=await self.connect();self.state.cast_origin.tcp_limit=2
        await self.connect()  # sockets with no complete HTTP request are not idle keep-alive
        rejected=await self.connect();code,h,body=await self.response(rejected)
        self.assertEqual(code,503);self.assertEqual(h['x-cast-reject-reason'],'tcp-connection-cap')
        self.assertEqual(len(self.state.connections),2)

    async def test_first_headers_have_absolute_timeout(self):
        origin.HEADER_TIMEOUT=.15
        pair=await self.connect()
        pair[1].write(b'GET /healthz HTTP/1.1\r\nHost: ');await pair[1].drain()
        await asyncio.sleep(.07);pair[1].write(b'partial');await pair[1].drain()
        self.assertEqual(await asyncio.wait_for(pair[0].read(),1),b'')
        self.assertEqual(self.state.cast_origin.header_expired,1)

    async def test_active_disconnect_releases_counter(self):
        pair=await self.connect();await self.send(pair,self.path);await self.wait_started(1)
        pair[1].close();await pair[1].wait_closed();self.gate.set()
        await asyncio.sleep(.05)
        self.assertEqual(self.state.cast_origin.active,0)


if __name__=='__main__':unittest.main()
