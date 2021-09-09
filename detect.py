"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync
import cv2
import os
import threading
import datetime
import time
import argparse
import socket as Socket
import re
import socket
import socketserver

from http.server import BaseHTTPRequestHandler, HTTPServer


# Define the thread that will continuously pull frames from the camera
class CameraBufferCleanerThread(threading.Thread):
    def __init__(self, camera, name='camera-buffer-cleaner-thread'):
        self.camera = camera
        self.last_frame = None
        self.last_frame_id = 0
        # self.FPS = 1/30
        # self.FPS_MS = int(self.FPS * 1000)
        super(CameraBufferCleanerThread, self).__init__(name=name)
        self.start()

    def run(self):
        while True:
            with lock:
                self.last_frame_id += 1
                _, self.last_frame = self.camera.read()
            # time.sleep(self.FPS)


camera = None
cam_cleaner = None
lock = threading.Lock()


def getIP(iface):
    search_str = 'ip addr show eth1'.format(iface)  # desktop: enp2s0
    ipv4 = re.search(re.compile(r'(?<=inet )(.*)(?=\/)', re.M), os.popen(search_str).read()).groups()[0]
    ipv6 = re.search(re.compile(r'(?<=inet6 )(.*)(?=\/)', re.M), os.popen(search_str).read()).groups()[0]
    return (ipv4, ipv6)


def compress(orig, comp):
    return float(orig) / float(comp)


class mjpgServer(BaseHTTPRequestHandler):
    """
    A simple mjpeg server that either publishes images directly from a camera
    or republishes images from another pygecko process.
    """

    ip = None
    hostname = None
    def do_GET(self):
        global camera
        global cam_cleaner
        global model
        global device
        conf_thres = 0.4
        iou_thres = 0.5
        print('connection from:', self.address_string())
        if self.ip is None or self.hostname is None:
            self.ip, _ = getIP('eth1')  # desktop: enp2s0
            self.hostname = Socket.gethostname()
        if self.path == '/mjpg':
            self.send_response(200)
            self.send_header(
                'Content-type',
                'multipart/x-mixed-replace; boundary=--jpgboundary'
            )
            self.end_headers()
            last_frame_id = -1
            while True:
                if camera:
                    pass
                else:
                    raise Exception('Error, camera not setup')

                if cam_cleaner.last_frame is not None:
                    if last_frame_id is cam_cleaner.last_frame_id:
                        # wait for next frame
                        # time.sleep(1/30)
                        continue
                    last_frame_id = cam_cleaner.last_frame_id
                    frames = cam_cleaner.last_frame
                    dataset = LoadStreams(frames, img_size=imgsz)
                    im0s = img.copy()
                    img = torch.from_numpy(img).to(device)
                    if len(img.shape) == 3:
                        img = img[None]
                    pred = model(img, augment=False, visualize=False)[0]
                    pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False, classes=None, max_det=1000)
                    # Process predictions
                    for i, det in enumerate(pred):  # detections per image
                        im0, frame = im0s.copy(), getattr(dataset, 'frame', 0)
                        if len(det):
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                c = int(cls)  # integer class
                                plot_one_box(xyxy, im0, label=f'{conf * 100:.0f}%', color=colors(c, True), line_thickness=3)
                    resized = cv2.resize(im0, (800, 450), interpolation=cv2.INTER_AREA)
                    ret, jpg = cv2.imencode('.jpg', resized,[cv2.IMWRITE_JPEG_QUALITY, 50, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
                    # print 'Compression ratio: %d4.0:1'%(compress(img.size,jpg.size))
                    self.wfile.write("--jpgboundary\r\n".encode("utf-8"))
                    self.send_header('Content-type', 'image/jpeg')
                    # self.send_header('Content-length',str(tmpFile.len))
                    self.send_header('Content-length', str(jpg.size))
                    self.end_headers()
                    # self.wfile.write("\n")
                    self.wfile.write(jpg.tostring())
                    # time.sleep(0.05) #20fps
                    # cv2.imshow('The last frame', resized)
                else:
                    print('no image from camera')
                    time.sleep(1)
                    continue

        elif self.path == '/':
            # hn = self.server.server_address[0]
            port = self.server.server_address[1]
            ip = self.ip
            hostname = self.hostname

            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write('<html><head></head><body>'.encode("utf-8"))
            self.wfile.write('<h1>{0!s}[{1!s}]:{2!s}</h1>'.format(hostname, ip, port).encode("utf-8"))
            self.wfile.write('<img src="http://{}:{}/mjpg"/>'.format(ip, port).encode("utf-8"))
            self.wfile.write('<p>{0!s}</p>'.format((self.version_string())).encode("utf-8"))
            # self.wfile.write('<p>The mjpg stream can be accessed directly at:<ul>')
            # self.wfile.write('<li>http://{0!s}:{1!s}/mjpg</li>'.format(ip, port))
            # self.wfile.write('<li><a href="http://{0!s}:{1!s}/mjpg"/>http://{0!s}:{1!s}/mjpg</a></li>'.format(hostname, port))
            # self.wfile.write('</p></ul>')
            # self.wfile.write('<p>This only handles one connection at a time</p>'.encode("utf-8"))
            self.wfile.write('</body></html>'.encode("utf-8"))

        else:
            print('error', self.path)
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write('<html><head></head><body>')
            self.wfile.write('<h1>{0!s} not found</h1>'.format(self.path))
            self.wfile.write('</body></html>')


def handleArgs():
    parser = argparse.ArgumentParser(description='A simple mjpeg server Example: mjpeg-server -p 8080 --camera 4')
    parser.add_argument('-p', '--port', help='mjpeg publisher port, default is 9000', type=int, default=9000)
    parser.add_argument('-c', '--camera', help='set opencv camera number, ex. -c 1', type=int, default=0)
    parser.add_argument('-t', '--type', help='set camera type, either pi or cv, ex. -t pi', default='cv')
    parser.add_argument('-s', '--size', help='set size', nargs=2, type=int, default=(320, 240))

    args = vars(parser.parse_args())
    args['size'] = (args['size'][0], args['size'][1])
    return args
def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    # Start the camera
    global camera
    camera = cv2.VideoCapture(
        "rtsp://admin:12345@10.10.1.184:554/Streaming/Channels/101?transportmode=unicast&profile=Profile_1",
        cv2.CAP_FFMPEG)
    # camera.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    print(camera.set(cv2.VIDEO_ACCELERATION_ANY, 1))
    # Start the cleaning thread
    global cam_cleaner
    cam_cleaner = CameraBufferCleanerThread(camera)

    ipv4, ipv6 = getIP('eth1')
    print('eth1:', ipv4)
    port = 8056
    # server = HTTPServer((ipv4, port), mjpgServer)
    print("server started on {}:{}".format(Socket.gethostname(), port))
    # server.serve_forever()

    addr = ('', port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(addr)
    sock.listen(5)
    # Launch 100 listener threads.
    class Thread(threading.Thread):
        def __init__(self, i):
            threading.Thread.__init__(self)
            self.i = i
            self.daemon = True
            self.start()

        def run(self):
            mjpgServer.ip = ipv4
            mjpgServer.hostname = Socket.gethostname()
            httpd = HTTPServer((ipv4, port), mjpgServer, False)

            # Prevent the HTTP server from re-binding every handler.
            # https://stackoverflow.com/questions/46210672/
            httpd.socket = sock
            httpd.server_bind = self.server_close = lambda self: None

            httpd.serve_forever()

    [Thread(i) for i in range(10)]


if __name__ == "__main__":
    global model
    global device
    webcam = True
    weights = 'yolov5s.pt'
    imgsz = 640
    # Initialize
    set_logging()
    device = select_device(0)
    # Load model
    opt = handleArgs()
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    cudnn.benchmark = True  # set True to speed up constant image size inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    main(opt)
