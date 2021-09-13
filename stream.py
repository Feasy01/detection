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
import numpy as np
from pathlib import Path
import sys

from http.server import BaseHTTPRequestHandler, HTTPServer

from numpy.lib.type_check import imag

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())

class stream:
    def __init__(self) -> None:
        try:
            ipv4, ipv6 = getIP('eth1')
            print('eth1:', ipv4)
            port = 8058
            print("server started on {}:{}".format(Socket.gethostname(), port))

            addr = ('', port)
            sock = socket.socket (socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(addr)
            sock.listen(5)
            rgb = np.random.randint(255, size=(900,800,3),dtype=np.uint8)
            self.img = rgb
            global streamer_img
            streamer_img = rgb
            global streamer_img_frame
            streamer_img_frame = 0
            # Launch 10 listener threads.
            class Thread(threading.Thread):
                def __init__(self, i):
                    threading.Thread.__init__(self)
                    self.i = i
                    self.daemon = True
                    self.start()
                def run(self):
                    mjpgServer.ip = ipv4                
                    mjpgServer.hostname = Socket.gethostname()
                    # mjpgServer.img = self.img
                    httpd = HTTPServer((ipv4, port), mjpgServer, False)
                    # Prevent the HTTP server from re-binding every handler.
                    # https://stackoverflow.com/questions/46210672/
                    httpd.socket = sock
                    httpd.server_bind = self.server_close = lambda self: None
                    httpd.serve_forever()
            [Thread(i) for i in range(10)] 

        except KeyboardInterrupt:
            print('KeyboardInterrupt')

    def setNewImg(self, img=None):
        if img is None:
            img = np.random.randint(255, size=(900,800,3),dtype=np.uint8)
        global streamer_img
        global streamer_img_frame
        streamer_img = img
        streamer_img_frame += 1




def getIP(iface):
    search_str = 'ip addr show eth1'.format(iface) # desktop: enp2s0
    ipv4 = re.search(re.compile(r'(?<=inet )(.*)(?=\/)', re.M), os.popen(search_str).read()).groups()[0]
    ipv6 = re.search(re.compile(r'(?<=inet6 )(.*)(?=\/)', re.M), os.popen(search_str).read()).groups()[0]
    return (ipv4, ipv6)

class mjpgServer(BaseHTTPRequestHandler):
    """
    A simple mjpeg server that either publishes images directly from a camera
    or republishes images from another pygecko process.
    """
    ip = None
    hostname = None
    img = None
    global streamer_img
    global streamer_img_frame

    def do_GET(self):
        print('connection from:', self.address_string())

        if self.ip is None or self.hostname is None:
            self.ip, _ = getIP('eth1') # desktop: enp2s0
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
                if streamer_img is not None:
                    if streamer_img_frame == last_frame_id:
                        continue
                    last_frame_id = streamer_img_frame
                    timestamp = datetime.datetime.now()
                    frame = streamer_img
                    resized = cv2.resize(frame, (800, 450), interpolation = cv2.INTER_AREA)
                    # resized = cv2.resize(frame, (400, 225), interpolation = cv2.INTER_AREA)
                    cv2.putText(resized, 
                        timestamp.strftime("%d %m %Y %I:%M:%S.%f"), 
                        (10, resized.shape[0] - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                    # cv2.imshow('The last frame', resized)
                else: 
                    print('no image from camera')
                    time.sleep(1)
                    continue

                ret, jpg = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 50, 
                                                          cv2.IMWRITE_JPEG_OPTIMIZE, 1])
                                                          
                # print 'Compression ratio: %d4.0:1'%(compress(img.size,jpg.size))
                self.wfile.write("--jpgboundary\r\n".encode("utf-8"))
                self.send_header('Content-type', 'image/jpeg')
                # self.send_header('Content-length',str(tmpFile.len))
                self.send_header('Content-length', str(jpg.size))
                self.end_headers()
                # self.wfile.write("\n")
                self.wfile.write(jpg.tostring())
                # time.sleep(0.05) #20fps

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





def main():
    myStream = stream()
    for i in range(200):
        myStream.setNewImg()
        time.sleep(0.5)
    print("Close")
    time.sleep(100)



if __name__ == '__main__':
    print("Starting")
    main()
