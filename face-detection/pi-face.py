# only_my_face_headless_win.py
# Windows headless "only my face": terminal messages + optional LiveKit dispatch.
# Robust Windows camera opening (DSHOW first), InsightFace background worker, no GUI.

import os, sys, time, pathlib, threading, queue, datetime, subprocess
import numpy as np
import cv2
from dotenv import load_dotenv

# ---- Agent Execution (optional) ----
LIVEKIT_ENABLED = True  # set False if you don't want to launch the agent
AGENT_NAME = os.getenv("AGENT_NAME", "face-detection-assistant")
AGENT_PROCESS = None # Global variable to hold the running agent process

# Load environment variables (edit path if your .env lives elsewhere)
load_dotenv(dotenv_path="../.env.local")

# ---- InsightFace ----
from insightface.app import FaceAnalysis

# ======== Tunables ========
SIM_THRESH          = 0.48      # stricter ~0.50, looser ~0.45
DETECT_INTERVAL     = 12        # heavy detect/embedding every N frames (bigger=faster)
DET_FRAME_SIZE      = (224, 224)# smaller = faster (192/224/256)
REQUEST_FPS         = 30
CAPTURE_SIZE        = (640, 480)
POS_STREAK          = 2         # positives required to declare "present"
NEG_STREAK          = 6         # misses to declare "absent"
HEARTBEAT_SEC       = 5
ABSENT_TIMEOUT_SEC  = 10 * 60   # after 10 minutes away, next detection triggers dispatch
EMB_MEAN_PATH       = "my_face_arcface.npy"
EMB_TEMPLATES_PATH  = "my_face_arcface_templates.npy"  # optional, if you saved templates
# ==========================

def now_str():
    return datetime.datetime.now().strftime("%H:%M:%S")

# ---------- Windows camera open helper ----------
# Prefers DirectShow (DSHOW) on Windows, then MSMF, then CAP_ANY.
# Tries MJPG first; if first read fails, falls back to YUY2 automatically.
def try_open_win_cam(preferred_indices=range(12), width=640, height=480, fps=30):
    def _configure(cap):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS,          fps)

    # 1) DirectShow
    for idx in preferred_indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            _configure(cap)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUY2'))
                ok, frame = cap.read()
            if ok and frame is not None:
                return cap, idx, 'DSHOW'
            cap.release()

    # 2) MSMF
    for idx in preferred_indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
        if cap.isOpened():
            _configure(cap)
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap, idx, 'MSMF'
            cap.release()

    # 3) CAP_ANY fallback
    for idx in preferred_indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            _configure(cap)
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap, idx, 'ANY'
            cap.release()

    return None, None, None

# ---------- Background worker: InsightFace on downscaled frames ----------
class DetectorWorker(threading.Thread):
    def __init__(self, det_size):
        super().__init__(daemon=True)
        self.q = queue.Queue(maxsize=1)
        self.out = queue.Queue(maxsize=1)
        self.stop_flag = False
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=-1, det_size=det_size)  # CPU

    def submit(self, frame_small):
        if not self.q.full():
            self.q.put(frame_small)

    def run(self):
        while not self.stop_flag:
            try:
                frame_small = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            faces = self.app.get(frame_small)
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])) if faces else None
            if face is None or face.normed_embedding is None:
                if not self.out.full():
                    self.out.put(None)
                continue
            emb = face.normed_embedding.astype(np.float32)
            if not self.out.full():
                self.out.put(emb)

# ---------- Agent Execution (Console Mode) ----------
def launch_agent():
    global AGENT_PROCESS
    if not LIVEKIT_ENABLED:
        print(f"[{now_str()}] ⚠️ Agent launch disabled.")
        return
    
    if AGENT_PROCESS and AGENT_PROCESS.poll() is None:
        print(f"[{now_str()}] ℹ️ Agent is already running.")
        return

    try:
        # Command to activate conda environment and run the agent in console mode
        # We use 'cmd /k' to keep the window open after execution (optional, but helpful for debugging)
        # We use '&&' to chain commands.
        # We change directory to D:\rasberry first to ensure agent.py and .env.local are found.
        
        # Note: The user is running from D:\rasberry\face-detection, so we need to go up one level.
        agent_dir = os.path.join(os.path.dirname(__file__), '..')
        
        # Use 'start cmd /K' to reliably open a new console window and keep it open.
        # We use 'call' to ensure conda activate runs properly in the new shell.
        # We must use shell=True for the 'start' command to be recognized.
        command = f'start cmd /K "call conda activate facerasberry && cd /d "{agent_dir}" && python agent.py console"'
        
        # Launch in a new console window (Windows specific)
        AGENT_PROCESS = subprocess.Popen(
            command,
            shell=True,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        print(f"[{now_str()}] 🚀 Agent launched in new console (PID: {AGENT_PROCESS.pid}).")
    except Exception as e:
        print(f"[{now_str()}] ❌ Agent launch failed: {e}")

def load_embeddings():
    mean_path = pathlib.Path(EMB_MEAN_PATH)
    if not mean_path.exists():
        sys.exit(f"Embedding file not found: {mean_path}. Run your enrollment script first.")
    mean_emb = np.load(mean_path).astype(np.float32)
    if mean_emb.shape != (512,):
        sys.exit("Bad mean embedding shape; re-enroll.")

    tmpls_path = pathlib.Path(EMB_TEMPLATES_PATH)
    if tmpls_path.exists():
        tmpls = np.load(tmpls_path).astype(np.float32)
        if tmpls.ndim == 2 and tmpls.shape[1] == 512:
            # ensure normalized
            norms = np.maximum(np.linalg.norm(tmpls, axis=1, keepdims=True), 1e-9)
            tmpls = tmpls / norms
        else:
            tmpls = None
    else:
        tmpls = None
    return mean_emb, tmpls

def main():
    # Try to keep OpenCV thread usage modest
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    # Load enrolled embeddings
    my_emb, my_templates = load_embeddings()

    # Open camera on Windows
    cap, cam_index, backend = try_open_win_cam(width=CAPTURE_SIZE[0], height=CAPTURE_SIZE[1], fps=REQUEST_FPS)
    if cap is None:
        sys.exit("No camera opened. Check Windows privacy settings, drivers, and close other apps using the camera.")
    print(f"[{now_str()}] ✅ Camera opened: index={cam_index} backend={backend}")

    worker = DetectorWorker(det_size=DET_FRAME_SIZE)
    worker.start()

    present = False
    pos_hits = 0
    neg_hits = 0
    last_heartbeat = time.time()

    # dispatch state
    last_absent_time = time.time()
    dispatch_ready = True # Start ready to dispatch on first detection

    frame_i = 0
    t0 = time.time(); frames = 0

    print("Windows headless mode. Press Ctrl+C to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break

            # Every N frames, send a small copy to worker
            if (frame_i % DETECT_INTERVAL) == 0:
                small = cv2.resize(frame, DET_FRAME_SIZE, interpolation=cv2.INTER_LINEAR)
                worker.submit(small)

            # Check for embedding result
            try:
                emb = worker.out.get_nowait()
                if emb is None:
                    pos_hits = 0
                    neg_hits = min(NEG_STREAK, neg_hits + 1)
                else:
                    # cosine similarity: use templates if available, else mean
                    if my_templates is not None:
                        sim = float(np.max(my_templates @ emb))
                    else:
                        sim = float(np.dot(emb, my_emb))
                    if sim >= SIM_THRESH:
                        pos_hits = min(POS_STREAK, pos_hits + 1)
                        neg_hits = 0
                    else:
                        pos_hits = 0
                        neg_hits = min(NEG_STREAK, neg_hits + 1)
            except queue.Empty:
                pass

            # state transitions
            if not present and pos_hits >= POS_STREAK:
                present = True
                print(f"[{now_str()}] ✅ YOUR FACE DETECTED")
                if dispatch_ready:
                    print(f"[{now_str()}] ⏳ Face detected after long absence. Launching agent in console mode…")
                    launch_agent()
                    dispatch_ready = False

            elif present and neg_hits >= NEG_STREAK:
                present = False
                print(f"[{now_str()}] ❌ Lost your face")
                last_absent_time = time.time()

            # arm dispatch after long absence
            if not present and not dispatch_ready and (time.time() - last_absent_time) >= ABSENT_TIMEOUT_SEC:
                dispatch_ready = True
                print(f"[{now_str()}] 💤 Absent for > {ABSENT_TIMEOUT_SEC}s. Agent will dispatch on next detection.")

            # heartbeat
            if time.time() - last_heartbeat >= HEARTBEAT_SEC:
                elapsed = time.time() - t0
                fps = frames / elapsed if elapsed > 0 else 0.0
                state = "present" if present else "absent"
                dispatch_status = "READY" if dispatch_ready else "IDLE"
                print(f"[{now_str()}] ~{fps:.1f} FPS | state: {state} | dispatch: {dispatch_status}")
                t0 = time.time(); frames = 0
                last_heartbeat = time.time()

            frame_i += 1
            frames += 1
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nStopping...")

    worker.stop_flag = True
    cap.release()

if __name__ == "__main__":
    main()
