import asyncio
import unittest
from fastapi import HTTPException
from ordered_pipeline import OrderedPipeline


async def receive(value, delay=0):
    await asyncio.sleep(delay)
    return value


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.processed = []
        self.gpu_count = self.max_gpu = 0
        self.gpu_gate = asyncio.Event()
        self.gpu_gate.set()
        async def process(sid, seq, decoded, timestamp):
            self.gpu_count += 1
            self.max_gpu = max(self.max_gpu, self.gpu_count)
            await self.gpu_gate.wait()
            await asyncio.sleep(.01)
            self.gpu_count -= 1
            if decoded == b"fail":
                raise HTTPException(500, "fake GPU failure")
            self.processed.append(seq)
            return {"seq":seq, "timing_ms": {}}
        async def decode(payload):
            await asyncio.sleep(.002)
            if payload == b"bad":
                raise HTTPException(422, "bad image")
            return payload
        def validate(sid):
            if sid != "test":
                raise HTTPException(409)
        self.p = OrderedPipeline(process, decode, validate, capacity=4,
                                 gap_timeout=.1, upload_timeout=.15)
        self.epoch = self.p.bind("test")

    async def asyncTearDown(self):
        self.gpu_gate.set()
        await self.p.close()

    def task(self, seq, payload=b"ok", delay=0, epoch=None):
        return asyncio.create_task(self.p.handle("test", seq, lambda: receive(payload,delay), epoch=epoch))

    async def expect_error(self, task, status):
        with self.assertRaises(HTTPException) as ctx:
            await task
        self.assertEqual(ctx.exception.status_code, status)

    async def test_out_of_order_upload_and_decode(self):
        tasks=[self.task(2),self.task(0,delay=.04),self.task(1,delay=.02),self.task(3)]
        await asyncio.sleep(.008)
        self.assertEqual(len(self.p.jobs),4)
        self.assertEqual(self.processed,[])
        results=await asyncio.gather(*tasks)
        self.assertEqual([r['seq'] for r in results],[2,0,1,3])
        self.assertEqual(self.processed,[0,1,2,3])
        self.assertEqual(self.max_gpu,1)
        self.assertEqual(self.p.next_seq,4)

    async def test_admits_while_running(self):
        self.gpu_gate.clear()
        first=self.task(0)
        await asyncio.sleep(.02)
        second=self.task(1)
        # Do not depend on OS scheduling within an arbitrary 20ms wall clock.
        async def wait_ready():
            while 1 not in self.p.jobs or self.p.jobs[1].state!='ready':
                await asyncio.sleep(.002)
        await asyncio.wait_for(wait_ready(),1)
        self.assertEqual(self.p.jobs[0].state,'running')
        self.assertEqual(self.p.jobs[1].state,'ready')
        self.gpu_gate.set()
        await asyncio.gather(first,second)

    async def test_bounded_reject_does_not_read_body(self):
        self.gpu_gate.clear()
        tasks=[self.task(i) for i in range(4)]
        await asyncio.sleep(.02)
        called=[]
        async def body():
            called.append(True)
        await self.expect_error(self.p.handle('test',4,body),429)
        self.assertEqual(called,[])
        self.assertEqual(len(self.p.jobs),4)
        self.gpu_gate.set()
        await asyncio.gather(*tasks)
        self.assertEqual((await self.task(4))['seq'],4)

    async def test_duplicate_rejected(self):
        first=self.task(0,delay=.03)
        await asyncio.sleep(.005)
        await self.expect_error(self.task(0),409)
        await first
        await self.expect_error(self.task(0),409)

    async def test_missing_predecessor_timeout_and_retry(self):
        await self.expect_error(self.task(1),409)
        self.assertEqual(self.processed,[])
        self.assertEqual(self.p.jobs,{})
        await self.task(0)

    async def test_invalid_image_cancels_successors(self):
        tasks=[self.task(0,b'bad',delay=.04),self.task(1),self.task(2)]
        results=await asyncio.gather(*tasks,return_exceptions=True)
        self.assertEqual([r.status_code for r in results],[422,409,409])
        self.assertEqual(self.processed,[])
        self.assertFalse(self.p.failed)
        await self.task(0)

    async def test_upload_timeout(self):
        tasks=[self.task(0,delay=.4),self.task(1)]
        results=await asyncio.gather(*tasks,return_exceptions=True)
        self.assertEqual([r.status_code for r in results],[408,409])
        self.assertEqual(self.p.jobs,{})

    async def test_failed_gpu_requires_reset(self):
        tasks=[self.task(0,b'fail'),self.task(1)]
        results=await asyncio.gather(*tasks,return_exceptions=True)
        self.assertEqual([r.status_code for r in results],[500,409])
        await self.expect_error(self.task(0),409)
        async with self.p.control():
            self.p.bind('test')
        await self.task(0)

    async def test_cancelled_client_does_not_skip_frame(self):
        first=self.task(0,delay=.03)
        await asyncio.sleep(.005)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        second=await self.task(1)
        self.assertEqual(second['seq'],1)
        self.assertEqual(self.processed,[0,1])

    async def test_control_and_stale_epoch(self):
        first=self.task(0,delay=.04)
        await asyncio.sleep(.005)
        with self.assertRaises(HTTPException):
            async with self.p.control():
                pass
        await first
        async with self.p.control():
            self.p.bind('test')
        await self.expect_error(self.task(0,epoch=self.epoch),409)
        await self.task(0,epoch=self.p.epoch)

    async def test_slow_cancelled_ingest_retains_capacity(self):
        tasks=[self.task(0,b'bad',delay=.01),self.task(1,delay=.12)]
        await self.expect_error(tasks[0],422)
        self.assertEqual(len(self.p.jobs),1)
        self.assertEqual(self.p.jobs[1].state,'uploading')
        await self.expect_error(tasks[1],409)

    async def test_ten_frame_window(self):
        self.p.capacity=10
        self.p.gap_timeout=2
        self.gpu_gate.clear()
        tasks=[self.task(i) for i in reversed(range(10))]
        async def all_ready():
            while len(self.p.jobs)!=10 or any(j.state not in ('ready','running') for j in self.p.jobs.values()):
                await asyncio.sleep(.002)
        await asyncio.wait_for(all_ready(),1)
        self.assertEqual(len(self.p.jobs),10)
        await self.expect_error(self.task(10),429)
        self.gpu_gate.set()
        responses=await asyncio.gather(*tasks)
        self.assertEqual([r['seq'] for r in responses],list(reversed(range(10))))
        self.assertEqual(self.processed,list(range(10)))
        self.assertEqual(self.max_gpu,1)
        await self.task(10)


if __name__=='__main__':
    unittest.main()
