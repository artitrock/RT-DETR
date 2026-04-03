import torch
import torch.nn as nn 
import torchvision.transforms as T
from torch.cuda.amp import autocast
import numpy as np 
from PIL import Image, ImageDraw, ImageFont
import os 
import sys 

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS


# =========================
# ✅ draw (แก้ warning + output name)
# =========================
def draw(images, labels, boxes, scores, thrh=0.6, path=""):
    
    try:
        font = ImageFont.truetype("../../fonts/ARIAL.TTF", size=20)
    except OSError:
        font = ImageFont.load_default()  # fallback
        
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        # ❌ OLD
        # scr = scores[i]
        # lab = labels[i][scr > thrh]
        # box = boxes[i][scr > thrh]

        # ✅ NEW (detach ป้องกัน warning)
        scr = scores[i].detach().cpu()
        lab = labels[i].detach().cpu()
        box = boxes[i].detach().cpu()

        mask = scr > thrh
        lab = lab[mask]
        box = box[mask]
        scrs = scr[mask]

        for j, b in enumerate(box):
            # ❌ OLD
            # draw.rectangle(list(b), ...)

            # ✅ NEW (แปลงเป็น list ปกติ)
            b = b.tolist()

            draw.rectangle(b, outline='yellow', width=10)
            draw.text(
                (b[0], b[1]),
                text=f"label: {lab[j].item()} {round(scrs[j].item(),2)}",
                font=font,
                fill='blue'
            )

        # ❌ OLD
        # if path == "":
        #     im.save(f'results_{i}.jpg')

        # ✅ NEW (รองรับชื่อไฟล์)
        if path == "":
            im.save(f"results_{i}.jpg")
        else:
            im.save(path)


def main(args):
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume')

    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    model = Model().to(args.device)

    im_pil = Image.open(args.im_file).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(args.device)

    transforms = T.Compose([
        T.Resize((640, 640)),  
        T.ToTensor(),
    ])

    im_data = transforms(im_pil)[None].to(args.device)

    # =========================
    # ✅ ป้องกัน grad (สำคัญ)
    # =========================
    with torch.no_grad():

        if args.sliced:
            slices, coordinates = slice_image(im_pil, 640, 640, 0.2)
            predictions = []

            for slice_img in slices:
                slice_tensor = transforms(slice_img)[None].to(args.device)

                with autocast():
                    output = model(
                        slice_tensor,
                        torch.tensor([[slice_img.size[0], slice_img.size[1]]]).to(args.device)
                    )

                labels, boxes, scores = output

                # ❌ OLD
                # labels = labels.cpu().detach().numpy()

                # ✅ NEW
                labels = labels.detach().cpu().numpy()
                boxes = boxes.detach().cpu().numpy()
                scores = scores.detach().cpu().numpy()

                predictions.append((labels, boxes, scores))

        else:
            labels, boxes, scores = model(im_data, orig_size)

    # =========================
    # ✅ ใช้ output path
    # =========================
    draw([im_pil], labels, boxes, scores, 0.6, args.output)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-r', '--resume', type=str)
    parser.add_argument('-f', '--im-file', type=str)
    parser.add_argument('-s', '--sliced', type=bool, default=False)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-nc', '--numberofboxes', type=int, default=25)

    # ✅ NEW (เพิ่ม output)
    parser.add_argument('-o', '--output', type=str, default='output.jpg')

    args = parser.parse_args()
    main(args)
