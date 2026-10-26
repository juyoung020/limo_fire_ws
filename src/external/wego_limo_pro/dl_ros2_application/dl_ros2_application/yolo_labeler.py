import cv2
import os
import numpy as np
import tkinter as tk
import tkinter.filedialog

SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png')
FONT          = cv2.FONT_HERSHEY_DUPLEX
FONT_BOLD     = cv2.FONT_HERSHEY_DUPLEX
HUD_HEIGHT    = 60
ROW_HEIGHT    = 26
HEAD_HEIGHT   = 26
PADDING       = 16
HUD_BG_COLOR  = (245, 245, 245)
TABLE_BG      = (240, 240, 240)
HEADER_TXT    = (20, 20, 20)
ACTION_TXT    = (50, 50, 50)
SHADOW_COLOR  = (200, 200, 200)
ALPHA_HUD     = 0.95
ALPHA_TABLE   = 0.95
ALPHA_LABELBG = 0.85
KEY_QUIT   = ord('q'); KEY_SAVE = ord('s'); KEY_DELETE = ord('d')
KEY_NEXT   = ord('n'); KEY_PREV = ord('p'); KEY_BASE = ord('0')
KEY_LEFT   = 65361
KEY_RIGHT  = 65363

class YOLOLabeler:
    def __init__(self):
        print("Guide: drag=box, 0-9=class, S=save, D=delete, N=next, p=prev, Q=quit")
        self.image_dir  = self._ask_dir("Select Images Directory")
        self.output_dir = self._ask_dir("Select Labels Directory")
        self._prepare_dirs()
        self.class_names = self._ask_classes()
        self.colors = self._get_colors(len(self.class_names))
        self.image_files = [f for f in os.listdir(self.image_dir)
                            if f.lower().endswith(SUPPORTED_EXTENSIONS)]
        self.current_index = 0
        self.current_class_id = 0
        self.image_history = {}
        self.rectangles = []
        self.drawing = False
        self.start_x = self.start_y = -1
        self.controls = [("Left drag","Left click and drag to draw a box"),
                         ("0 ~ 9","Press a number key to pick a class"),
                         ("S","Press 'S' to save boxes in YOLO format"),
                         ("D","Press 'D' to remove the last box"),
                         ("N","Press 'N' to view the next image"),
                         ("P","Press 'P' to view the previous image"),
                         ("Q","Press 'Q' to quit the labeler")]

    def _get_colors(self, num_classes):
        """Generates a list of random colors for each class."""
        np.random.seed(42)  # For reproducible colors
        return [tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(num_classes)]

    def _ask_dir(self, title):
        root=tk.Tk(); root.withdraw()
        path=tk.filedialog.askdirectory(title=title, initialdir=os.path.expanduser("~"))
        root.destroy()
        return path

    def _prepare_dirs(self):
        self.labels_dir = os.path.join(self.output_dir,'labels')
        self.train_dir  = os.path.join(self.labels_dir,'train')
        self.val_dir    = os.path.join(self.labels_dir,'val')
        for d in (self.labels_dir,self.train_dir,self.val_dir):
            os.makedirs(d, exist_ok=True)

    def _ask_classes(self):
        while True:
            names=input("Class names (space-separated): ").split()
            if names: return names

    def _load_yolo_labels(self, name, shape):
        txt=os.path.join(self.train_dir, os.path.splitext(name)[0]+'.txt')
        h,w=shape[:2]; boxes=[]
        if os.path.exists(txt):
            for ln in open(txt):
                cid,xc,yc,wn,hn=map(float,ln.split())
                cid=int(cid); bw, bh = wn*w, hn*h
                x1=int(xc*w-bw/2); y1=int(yc*h-bh/2)
                boxes.append((x1,y1,x1+int(bw),y1+int(bh),cid))
        return boxes

    def _save_yolo_format(self, name, out_dir, rects, shape):
        h,w=shape[:2]; txt=os.path.join(out_dir, os.path.splitext(name)[0]+'.txt')
        with open(txt,'w') as f:
            for x1,y1,x2,y2,c in rects:
                xc=(x1+x2)/2/w; yc=(y1+y2)/2/h
                wn=abs(x2-x1)/w; hn=abs(y2-y1)/h
                f.write(f"{c} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

    def _build_canvas(self, img):
        ih, iw = img.shape[:2]
        table_h = HEAD_HEIGHT + len(self.controls)*ROW_HEIGHT
        canvas_h = HUD_HEIGHT + ih + table_h
        canvas = np.full((canvas_h, iw, 3), 255, np.uint8)

        header = canvas.copy()
        cv2.rectangle(header,(0,0),(iw,HUD_HEIGHT),HUD_BG_COLOR,-1)
        cv2.line(header, (0, HUD_HEIGHT), (iw, HUD_HEIGHT), SHADOW_COLOR, 1)
        cv2.line(header, (0, HUD_HEIGHT//2), (iw, HUD_HEIGHT//2), SHADOW_COLOR, 1)
        cv2.addWeighted(header, ALPHA_HUD, canvas,1-ALPHA_HUD, 0, canvas)
        class_info = f"Class: {self.class_names[self.current_class_id]} ({self.current_class_id})key"
        cv2.putText(canvas, class_info, (PADDING, 20), FONT, 0.55, HEADER_TXT, 1, cv2.LINE_AA)
        progress_info = f"Progress: {self.current_index+1} / {len(self.image_files)}"
        (tw, _), _ = cv2.getTextSize(progress_info, FONT, 0.55, 1)
        cv2.putText(canvas, progress_info, (iw-PADDING-tw, 20), FONT, 0.55, HEADER_TXT, 1, cv2.LINE_AA)
        dir_info = f"Dir: {self.image_dir}"
        (tw, _), _ = cv2.getTextSize(dir_info, FONT, 0.55, 1)
        cv2.putText(canvas, dir_info, (PADDING, HUD_HEIGHT-12), FONT, 0.55, HEADER_TXT, 1, cv2.LINE_AA)

        canvas[HUD_HEIGHT:HUD_HEIGHT+ih, 0:iw] = img

        y0=HUD_HEIGHT+ih
        footer = canvas.copy()
        cv2.rectangle(footer,(0,y0),(iw,canvas_h),TABLE_BG,-1)
        cv2.addWeighted(footer, ALPHA_TABLE, canvas,1-ALPHA_TABLE,0,canvas)
        cv2.rectangle(canvas, (0, y0), (iw, y0+HEAD_HEIGHT), TABLE_BG, -1)
        cv2.line(canvas, (0, y0), (iw, y0), SHADOW_COLOR, 1)
        cv2.putText(canvas,"Key",(PADDING,y0+17),FONT_BOLD,0.55,HEADER_TXT,1,cv2.LINE_AA)
        cv2.putText(canvas,"Action",(PADDING+110,y0+17),FONT_BOLD,0.55,HEADER_TXT,1,cv2.LINE_AA)
        cv2.line(canvas, (0, y0+HEAD_HEIGHT), (iw, y0+HEAD_HEIGHT), SHADOW_COLOR, 1)
        cv2.line(canvas,(PADDING+100,y0),(PADDING+100,canvas_h),SHADOW_COLOR,1)
        for i,(k,a) in enumerate(self.controls):
            yy=y0+HEAD_HEIGHT+(i+1)*ROW_HEIGHT-6
            cv2.putText(canvas,k,(PADDING,yy),FONT,0.55,ACTION_TXT,1,cv2.LINE_AA)
            cv2.putText(canvas,a,(PADDING+110,yy),FONT,0.55,ACTION_TXT,1,cv2.LINE_AA)
            cv2.line(canvas, (0, yy+10), (iw, yy+10), SHADOW_COLOR, 1)
        return canvas, HUD_HEIGHT

    def _redraw(self):
        base = self.original.copy()
        for x1,y1,x2,y2,c in self.rectangles:
            color = self.colors[c]
            cv2.rectangle(base,(x1,y1),(x2,y2),color,2)
            lbl=f"{self.class_names[c]} ({c})"
            (tw,th),_ = cv2.getTextSize(lbl,FONT,0.55,1)
            cv2.rectangle(base,(x1,y1-th-6),(x1+tw+4,y1),color,-1)
            cv2.addWeighted(base, ALPHA_LABELBG, base,1-ALPHA_LABELBG,0,base)
            cv2.putText(base,lbl,(x1+2,y1-2),FONT,0.55,(0,0,0),1,cv2.LINE_AA)
        canvas, self.y_offset = self._build_canvas(base)
        cv2.imshow("Labeling Image", canvas)
        self.display = canvas

    def _draw_existing_boxes(self, base):
        for x1,y1,x2,y2,c in self.rectangles:
            color = self.colors[c]
            cv2.rectangle(base,(x1,y1),(x2,y2),color,2)
            lbl=f"{self.class_names[c]} ({c})"
            (tw,th),_ = cv2.getTextSize(lbl,FONT,0.55,1)
            cv2.rectangle(base,(x1,y1-th-6),(x1+tw+4,y1),color,-1)
            cv2.addWeighted(base, ALPHA_LABELBG, base,1-ALPHA_LABELBG,0,base)
            cv2.putText(base,lbl,(x1+2,y1-2),FONT,0.55,(0,0,0),1,cv2.LINE_AA)
        return base

    def _mouse_cb(self, event, x, y, flags, param):
        if not hasattr(self, 'y_offset'):
            return
        if y < self.y_offset:
            return
        img_y = y - self.y_offset
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x, self.start_y = x, img_y
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            tmp = self.original.copy()
            tmp = self._draw_existing_boxes(tmp)
            cv2.rectangle(tmp, (self.start_x, self.start_y), (x, img_y), self.colors[self.current_class_id], 2)
            label = f"{self.class_names[self.current_class_id]} ({self.current_class_id})"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 1)
            cv2.rectangle(tmp, (self.start_x, self.start_y-th-6), (self.start_x+tw+4, self.start_y), self.colors[self.current_class_id], -1)
            cv2.addWeighted(tmp, ALPHA_LABELBG, tmp, 1-ALPHA_LABELBG, 0, tmp)
            cv2.putText(tmp, label, (self.start_x+2, self.start_y-2), FONT, 0.55, (0,0,0), 1, cv2.LINE_AA)
            preview, _ = self._build_canvas(tmp)
            cv2.imshow("Labeling Image", preview)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            if abs(x - self.start_x) > 5 and abs(img_y - self.start_y) > 5:
                self.rectangles.append((self.start_x, self.start_y, x, img_y, self.current_class_id))
            self._redraw()

    def process(self):
        print(f"Found {len(self.image_files)} images")
        cv2.namedWindow("Labeling Image", cv2.WINDOW_GUI_NORMAL|cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback("Labeling Image", self._mouse_cb)
        while 0 <= self.current_index < len(self.image_files):
            name = self.image_files[self.current_index]
            self.original = cv2.imread(os.path.join(self.image_dir, name))
            if self.original is None:
                print(f"Cannot open {name}")
                self.current_index += 1
                continue
            self.rectangles = self.image_history.get(name, self._load_yolo_labels(name, self.original.shape))
            self._redraw()
            while True:
                k = cv2.waitKeyEx(0) & 0xFF
                if k == KEY_QUIT:
                    cv2.destroyAllWindows()
                    return
                elif k == KEY_SAVE:
                    for d in (self.train_dir, self.val_dir):
                        self._save_yolo_format(name, d, self.rectangles, self.original.shape)
                    self.image_history[name] = list(self.rectangles)
                    print("Saved", name)
                    break
                elif k == KEY_DELETE and self.rectangles:
                    self.rectangles.pop()
                    self._redraw()
                elif k == KEY_NEXT:
                    self.current_index += 1
                    break
                elif k == KEY_PREV:
                    self.current_index -= 1
                    break
                elif KEY_BASE <= k <= KEY_BASE + 9 and k - KEY_BASE < len(self.class_names):
                    self.current_class_id = k - KEY_BASE
                    self._redraw()
def main():
    try:
        labeler = YOLOLabeler()
        labeler.process()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
