"""Detection-guided SAM: fresh person detection per frame, conservative fallback.

Generic/animal mode continues to use the unchanged original SAM algorithm.
"""
import numpy as np
from inference.realtime_mask import CausalBoxSegmenter
from person_detector import iou


class PromptedBoxSegmenter(CausalBoxSegmenter):
    prompt_override=None

    def _prompt_box(self,width,height):
        if self.prompt_override is not None:
            return np.asarray(self.prompt_override,dtype=np.float32)
        return super()._prompt_box(width,height)

    def reset(self):
        super().reset()
        self.prompt_override=None


def area(box):
    return max(0,box[2]-box[0])*max(0,box[3]-box[1])


def union(a,b):
    return [min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])]


def conservative_crop(det,sam,width,height,padding=5):
    """Never replace a semantic person box with unrelated wall/fragment mask."""
    accepted=False
    if sam is not None:
        inter=max(0,min(det[2],sam[2])-max(det[0],sam[0]))*max(0,min(det[3],sam[3])-max(det[1],sam[1]))
        coverage=inter/max(area(det),1)
        ratio=area(sam)/max(area(det),1)
        accepted=(iou(det,sam)>=.5 and coverage>=.75 and .65<=ratio<=1.6)
    merged=union(det,sam) if accepted else det
    box=[max(0,int(np.floor(merged[0]))-padding),max(0,int(np.floor(merged[1]))-padding),
         min(width,int(np.ceil(merged[2]))+padding),min(height,int(np.ceil(merged[3]))+padding)]
    return box,accepted


class HumanTracker:
    def __init__(self,detector,segmenter):
        self.detector,self.segmenter=detector,segmenter
        self.reset()

    def reset(self):
        self.previous_detection=None
        self.visible=False
        self.seen_once=False
        self.segmenter.reset()

    def select(self,candidates):
        if self.previous_detection is not None:
            ranked=sorted(candidates,key=lambda x:iou(x['box'],self.previous_detection),reverse=True)
            if iou(ranked[0]['box'],self.previous_detection)>.15:return ranked[0]
        # One target: largest confident person; not persistent multi-person identity tracking.
        return max(candidates,key=lambda x:area(x['box'])*x['score'])

    def box(self,rgb):
        candidates=self.detector.detect(rgb)
        if not candidates:
            was_visible=self.visible
            self.visible=False;self.previous_detection=None
            self.segmenter.reset()
            return None,{'mode':'human','status':'lost' if self.seen_once else 'waiting_for_person',
                         'person_found':False,'motion_valid':False,'reacquired':False,
                         'lost_now':was_visible,'detector_score':None,'detector_bbox':None,
                         'sam_bbox':None,'crop_source':'none','detector_ms':self.detector.last_ms,
                         'sam_ms':0.0,'person_candidates':0}
        target=self.select(candidates);det=target['box']
        reacquired=not self.visible and self.seen_once
        first_valid=not self.seen_once
        self.segmenter.prompt_override=det
        sam=list(self.segmenter.box(rgb))
        crop,accepted=conservative_crop(det,sam,rgb.shape[1],rgb.shape[0])
        # The SAM class can return the prompt/previous box on an empty mask.
        # It never controls the next prompt: that ALWAYS comes from detection.
        self.previous_detection=det;self.visible=True;self.seen_once=True
        return crop,{'mode':'human','status':'tracking','person_found':True,'motion_valid':True,
                     'reacquired':reacquired,'first_valid':first_valid,'lost_now':False,
                     'detector_score':target['score'],'detector_bbox':[round(float(x),3) for x in det],
                     'sam_bbox':sam,'sam_box_accepted':accepted,
                     'crop_source':'detector_sam_union' if accepted else 'detector_fallback',
                     'detector_ms':self.detector.last_ms,'sam_ms':self.segmenter.last_seconds*1000,
                     'person_candidates':len(candidates)}
