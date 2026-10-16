import json
import os
os.environ["CAST_HISTORY_ENABLED"]="1"
from pathlib import Path
import tempfile
import unittest
import numpy as np
from human_tracking import HumanTracker,conservative_crop
from live_history import LiveHistory


class FakeDetector:
    last_ms=1
    candidates=[]
    def detect(self,rgb):return self.candidates


class FakeSam:
    last_seconds=.001
    prompt_override=None
    result=[0,0,1,1]
    def reset(self):self.prompt_override=None
    def box(self,rgb):return self.result


class TrackingTests(unittest.TestCase):
    def test_background_mask_cannot_replace_person(self):
        det=[200,10,500,710]
        box,ok=conservative_crop(det,[500,0,720,720],720,720)
        self.assertFalse(ok);self.assertEqual(box,[195,5,505,715])

    def test_fragment_cannot_crop_out_head_and_feet(self):
        box,ok=conservative_crop([200,10,500,710],[260,220,440,450],720,720)
        self.assertFalse(ok);self.assertEqual(box,[195,5,505,715])

    def test_plausible_sam_union_contains_detection(self):
        box,ok=conservative_crop([200,10,500,710],[185,0,508,715],720,720)
        self.assertTrue(ok);self.assertEqual(box,[180,0,513,720])

    def test_absence_and_reacquisition(self):
        det=FakeDetector();sam=FakeSam();t=HumanTracker(det,sam);rgb=np.zeros((720,720,3),dtype=np.uint8)
        box,s=t.box(rgb);self.assertIsNone(box);self.assertEqual(s['status'],'waiting_for_person')
        det.candidates=[{'box':[200,0,500,720],'score':.9}];sam.result=[205,2,505,718]
        box,s=t.box(rgb);self.assertTrue(s['first_valid']);self.assertFalse(s['reacquired'])
        self.assertEqual(sam.prompt_override,[200,0,500,720])
        det.candidates=[];box,s=t.box(rgb);self.assertIsNone(box);self.assertEqual(s['status'],'lost')
        self.assertIsNone(sam.prompt_override)
        det.candidates=[{'box':[300,0,600,720],'score':.9}]
        box,s=t.box(rgb);self.assertTrue(s['reacquired']);self.assertEqual(sam.prompt_override,[300,0,600,720])

    def test_multiperson_prefers_previous_overlap(self):
        d=FakeDetector();sam=FakeSam();t=HumanTracker(d,sam)
        t.previous_detection=[200,0,400,600]
        c=[{'box':[0,0,600,720],'score':.99},{'box':[205,0,405,600],'score':.8}]
        self.assertEqual(t.select(c),c[1])

    def test_no_person_history_has_no_fake_crop(self):
        with tempfile.TemporaryDirectory() as d:
            h=LiveHistory(d);stream=h.new_stream()
            h.submit({'stream':stream,'session_label':'test','ts':1e20,'seq':0,'timestamp_ms':None,
                      'rgb':np.zeros((32,32,3),dtype=np.uint8),'mask':None,'bbox':None,
                      'rig_id':'test','joints':2,'timing_ms':{'server_total':1},
                      'motion':{'rot6d':[[1,0,0,0,1,0]]*2},'motion_valid':False,
                      'tracking':{'status':'waiting_for_person'}})
            h.queue.join();h.close()
            data=json.loads((Path(d)/'live_history/index.json').read_text());r=data['records'][0]
            self.assertIsNone(r['bbox']);self.assertFalse(r['motion_valid'])
            self.assertEqual(set(r['available']),{'original','bbox'})
            self.assertFalse((Path(d)/'live_history'/r['id']/'crop.jpg').exists())


if __name__=='__main__':unittest.main()
