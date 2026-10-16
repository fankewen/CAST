"""COCO person detector for SAM prompting, using existing torchvision only.

Fixed-shape FP32 CUDA graph of SSDLite320/MobileNetV3-Large. Same backbone,
anchors, box coder and person logits as torchvision; only person-class NMS on CPU.
No remote model downloads in service startup.
"""
import time
import torch
import torch.nn.functional as F
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.image_list import ImageList

from settings import SETTINGS
DEFAULT_WEIGHTS=SETTINGS.person_checkpoint


def iou(a,b):
    x0=max(a[0],b[0]);y0=max(a[1],b[1]);x1=min(a[2],b[2]);y1=min(a[3],b[3])
    overlap=max(0,x1-x0)*max(0,y1-y0)
    return overlap/max(1e-8,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap)


class PersonDetector:
    @torch.inference_mode()
    def __init__(self,weights=DEFAULT_WEIGHTS,device='cuda:0',threshold=.55):
        self.device=torch.device(device)
        self.threshold=threshold
        self.model=ssdlite320_mobilenet_v3_large(weights=None,weights_backbone=None).eval().to(self.device)
        self.model.load_state_dict(torch.load(str(weights),map_location=self.device,weights_only=True))
        self.input=torch.zeros((1,3,320,320),device=self.device,dtype=torch.float32)
        features=list(self.model.backbone(self.input).values())
        self.anchors=self.model.anchor_generator(ImageList(self.input,[(320,320)]),features)[0]
        self.anchor_wh=self.anchors[:,2:]-self.anchors[:,:2]
        self.anchor_center=self.anchors[:,:2]+.5*self.anchor_wh
        side=torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for _ in range(3): self._forward()
        torch.cuda.current_stream(self.device).wait_stream(side)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.output=self._forward()
        self.last_ms=0

    def _forward(self):
        features=list(self.model.backbone(self.input).values())
        heads=self.model.head(features)
        # Equivalent to BoxCoder.decode_single, without constructing CPU scalar
        # tensors during capture (torchvision's implementation does that).
        delta=heads['bbox_regression'][0]
        wx,wy,ww,wh=self.model.box_coder.weights
        center=torch.stack((delta[:,0]/wx,delta[:,1]/wy),dim=1)*self.anchor_wh+self.anchor_center
        size=torch.exp(torch.stack((delta[:,2]/ww,delta[:,3]/wh),dim=1).clamp(max=self.model.box_coder.bbox_xform_clip))*self.anchor_wh
        boxes=torch.cat((center-.5*size,center+.5*size),dim=1).clamp(0,320)
        scores=heads['cls_logits'][0].softmax(-1)[:,1]
        score,index=scores.topk(100)
        return torch.cat((boxes[index],score[:,None]),dim=1)

    @torch.inference_mode()
    def detect(self,rgb):
        start=time.perf_counter();height,width=rgb.shape[:2]
        x=torch.from_numpy(rgb).to(self.device).permute(2,0,1)[None].float()
        # Torchvision ObjectDetection -> GeneralizedRCNNTransform for this SSD:
        # RGB /255, normalize(mean=.5,std=.5), fixed bilinear 320x320 resize.
        x=x.mul(2/255).sub(1)
        x=F.interpolate(x,size=(320,320),mode='bilinear',align_corners=False)
        self.input.copy_(x);self.graph.replay()
        raw=self.output.cpu().numpy()
        raw[:,[0,2]]*=width/320;raw[:,[1,3]]*=height/320
        kept=[]
        for row in raw:
            score=float(row[4]);box=row[:4].astype(float).tolist()
            if score<self.threshold:continue
            if box[2]-box[0]<8 or box[3]-box[1]<16:continue
            if any(iou(box,k['box'])>.55 for k in kept):continue
            kept.append({'box':box,'score':score})
            if len(kept)>=10:break
        self.last_ms=(time.perf_counter()-start)*1000
        return kept
