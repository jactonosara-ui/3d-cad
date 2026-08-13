import cv2
import numpy as np
import http.server
import socketserver
import threading
import json
import time
import webbrowser

PORT = 8000
HOST = "127.0.0.1"

# Global state for spatial telemetry
telemetry_data = {"x": 0.0, "y": 0.0, "scale": 1.0, "detected": False, "fps": 0}
data_lock = threading.Lock()

# HSV Color Boundaries for Bright Green (adjust if using another color)
LOWER_HSV = np.array([35, 100, 100])
UPPER_HSV = np.array([85, 255, 255])

# Smoothing state
smooth_x, smooth_y, smooth_scale = 0.0, 0.0, 1.0
alpha = 0.35
prev_time = time.time()


def process_frame_buffer(image_bytes):
    """Processes frame received from HTML5 browser stream through OpenCV."""
    global telemetry_data, smooth_x, smooth_y, smooth_scale, prev_time

    # Decode JPEG bytes from browser into OpenCV BGR Matrix
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return

    h, w, _ = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = False
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area > 300:
            detected = True
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Normalize coordinates to range [-1.0, 1.0]
                raw_x = round((cx - (w / 2)) / (w / 2), 3)
                raw_y = round(-((cy - (h / 2)) / (h / 2)), 3)
                raw_scale = round(np.clip(np.sqrt(area) / 20.0, 0.4, 3.0), 2)

                smooth_x += alpha * (raw_x - smooth_x)
                smooth_y += alpha * (raw_y - smooth_y)
                smooth_scale += alpha * (raw_scale - smooth_scale)

    curr_time = time.time()
    fps = int(1.0 / (curr_time - prev_time + 1e-6))
    prev_time = curr_time

    with data_lock:
        telemetry_data = {
            "x": round(smooth_x, 3),
            "y": round(smooth_y, 3),
            "scale": round(smooth_scale, 2),
            "detected": detected,
            "fps": fps,
        }


# --- FRONTEND (HTML5 Camera Capture + Three.js) ---
HTML_FRONTEND = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gesture 3D Object Lab</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: #0f172a;
            color: #f8fafc;
            font-family: system-ui, -apple-system, sans-serif;
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }
        header {
            padding: 10px 16px;
            background: #1e293b;
            border-bottom: 1px solid #334155;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        h1 { font-size: 1rem; color: #38bdf8; }
        .status-badge {
            font-size: 0.75rem;
            padding: 4px 8px;
            border-radius: 12px;
            background: #334155;
            color: #94a3b8;
        }
        .status-badge.active { background: #166534; color: #4ade80; }

        .workspace {
            display: flex;
            flex-direction: column;
            flex: 1;
            gap: 8px;
            padding: 8px;
        }

        .camera-panel {
            position: relative;
            height: 35%;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #334155;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        video {
            height: 100%;
            width: 100%;
            object-fit: contain;
            transform: scaleX(-1); /* Mirror camera */
        }
        .overlay-info {
            position: absolute;
            top: 8px;
            left: 8px;
            background: rgba(15, 23, 42, 0.85);
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            color: #38bdf8;
            font-family: monospace;
        }

        .canvas-panel {
            flex: 1;
            background: #020617;
            border-radius: 8px;
            position: relative;
            border: 1px solid #334155;
            overflow: hidden;
        }
        #threejs-canvas { width: 100%; height: 100%; display: block; }
    </style>
</head>
<body>
    <header>
        <h1>GESTURE 3D LAB (TERMUX)</h1>
        <div id="status" class="status-badge">Allow Camera Access...</div>
    </header>

    <div class="workspace">
        <!-- TOP: Mobile HTML5 WebRTC Camera Preview -->
        <div class="camera-panel">
            <div class="overlay-info" id="telemetry">X: 0.00 | Y: 0.00 | Scale: 1.00</div>
            <video id="webcam" autoplay playsinline muted></video>
            <canvas id="capture-canvas" style="display:none;"></canvas>
        </div>

        <!-- BOTTOM: 3D Virtual Object Canvas -->
        <div class="canvas-panel">
            <canvas id="threejs-canvas"></canvas>
        </div>
    </div>

    <script>
        // 1. WEBCAM CAPTURE & STREAM TO PYTHON BACKEND
        const video = document.getElementById('webcam');
        const captureCanvas = document.getElementById('capture-canvas');
        const captureCtx = captureCanvas.getContext('2d');

        navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: 320, height: 240 } })
            .then((stream) => {
                video.srcObject = stream;
                video.onloadedmetadata = () => {
                    captureCanvas.width = 320;
                    captureCanvas.height = 240;
                    startFrameStream();
                };
            })
            .catch((err) => {
                alert("Camera permission denied or unavailable: " + err);
            });

        function startFrameStream() {
            setInterval(() => {
                if (video.readyState === video.HAVE_ENOUGH_DATA) {
                    captureCtx.drawImage(video, 0, 0, 320, 240);
                    captureCanvas.toBlob((blob) => {
                        if (blob) {
                            fetch('/upload_frame', { method: 'POST', body: blob });
                        }
                    }, 'image/jpeg', 0.5);
                }
            }, 40); // Send ~25 FPS to Python for processing
        }

        // 2. THREE.JS 3D SCENE SETUP
        const container = document.querySelector('.canvas-panel');
        const canvas = document.getElementById('threejs-canvas');

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x020617);

        const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 1000);
        camera.position.z = 5;

        const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
        renderer.setSize(container.clientWidth, container.clientHeight);

        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const dirLight = new THREE.DirectionalLight(0x38bdf8, 1.2);
        dirLight.position.set(5, 10, 7);
        scene.add(dirLight);

        // 3D Metallic Object
        const geometry = new THREE.TorusKnotGeometry(0.8, 0.28, 128, 32);
        const material = new THREE.MeshStandardMaterial({ color: 0x0284c7, metalness: 0.8, roughness: 0.2 });
        const targetMesh = new THREE.Mesh(geometry, material);
        scene.add(targetMesh);

        window.addEventListener('resize', () => {
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        });

        // 3. LISTEN FOR PYTHON OPENCV TELEMETRY
        const statusEl = document.getElementById('status');
        const telemetryEl = document.getElementById('telemetry');
        let targetX = 0, targetY = 0, targetScale = 1;

        const eventSource = new EventSource('/events');
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.detected) {
                statusEl.textContent = 'Tracking Active';
                statusEl.classList.add('active');
                targetX = data.x * 2.5;
                targetY = data.y * 1.8;
                targetScale = data.scale;
                telemetryEl.textContent = `X: ${data.x.toFixed(2)} | Y: ${data.y.toFixed(2)} | Scale: ${data.scale.toFixed(2)} | FPS: ${data.fps}`;
            } else {
                statusEl.textContent = 'Searching for Green Marker...';
                statusEl.classList.remove('active');
            }
        };

        // 4. ANIMATION LOOP
        function animate() {
            requestAnimationFrame(animate);
            targetMesh.position.x += (targetX - targetMesh.position.x) * 0.2;
            targetMesh.position.y += (targetY - targetMesh.position.y) * 0.2;

            const nextScale = targetMesh.scale.x + (targetScale - targetMesh.scale.x) * 0.2;
            targetMesh.scale.set(nextScale, nextScale, nextScale);

            targetMesh.rotation.x += 0.01;
            targetMesh.rotation.y += 0.015;

            renderer.render(scene, camera);
        }
        animate();
    </script>
</body>
</html>
"""


# --- HTTP SERVER & ROUTES ---
class LabRequestHandler(http.server.BaseHTTPRequestHandler):

    def do_POST(self):
        if self.path == "/upload_frame":
            content_length = int(self.headers["Content-Length"])
            image_bytes = self.rfile.read(content_length)
            process_frame_buffer(image_bytes)
            self.send_response(200)
            self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_FRONTEND.encode("utf-8"))

        elif self.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            while True:
                with data_lock:
                    payload = json.dumps(telemetry_data)
                try:
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except Exception:
                    break
                time.sleep(0.02)

    def log_message(self, format, *args):
        return


def main():
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer((HOST, PORT), LabRequestHandler)
    url = f"http://{HOST}:{PORT}"

    print("=" * 60)
    print(f" GESTURE 3D LAB (TERMUX MODE) RUNNING AT: {url}")
    print("=" * 60)

    webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shut down cleanly.")
        httpd.server_close()


if __name__ == "__main__":
    main()
