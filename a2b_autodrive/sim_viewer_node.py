#!/usr/bin/env python3
"""
sim_viewer_node  -  live top-down view of the A->B simulation in a web browser (Flask).

Open http://localhost:5000 (from Windows too: WSL forwards localhost).

Shows the true truck pose and trail, the lidar-based estimate and its trail, the latest
lidar scan, the planned route, point B, planner state, speed, error and PWM duties.
Buttons publish /a2b/enable (start / stop); clicking the map publishes /goal_pose (new B).

Install Flask once:  sudo apt install python3-flask
"""
import logging
import math
import re
import threading
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Bool, Float32MultiArray, String
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from robominer_msgs.msg import LocalizationData

try:
    from flask import Flask, jsonify, request, Response
except ImportError:  # pragma: no cover
    Flask = None

BE = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)


class SimViewer(Node):
    def __init__(self):
        super().__init__("sim_viewer")
        self.declare_parameter("port", 5000)
        self.declare_parameter("truth_topic", "/sim/ground_truth")
        self.declare_parameter("estimate_topic", "/localization/localization_data")
        self.declare_parameter("points_topic", "/fake_lidar/points")
        self.declare_parameter("show_room", True)
        self.declare_parameter("lidar_x_offset", 0.25)
        self.declare_parameter("scan_max_points", 1500)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.port = int(g("port"))
        self.off = g("lidar_x_offset")
        self.max_pts = int(g("scan_max_points"))
        self.lock = threading.Lock()
        self.truth = None
        self.est = None
        self.truth_trail = deque(maxlen=4000)
        self.est_trail = deque(maxlen=4000)
        self.route = []
        self.status = "IDLE"
        self.reason = ""
        self.info = {}
        self.pwm = [7.5, 7.5, 7.5]
        self.scan = []
        self.goal = None
        self.t0 = time.monotonic()

        self.room = None
        if g("show_room"):
            try:
                from a2b_autodrive import fake_lidar_node as fl
                self.room = {"walls": [list(w[:4]) for w in fl.WALLS],
                             "boxes": [list(b[:4]) for b in fl.BOXES],
                             "pillars": [list(p[:3]) for p in fl.PILLARS]}
            except Exception:  # noqa: BLE001
                self.room = None

        self.create_subscription(LocalizationData, g("truth_topic"), self.on_truth, BE)
        if g("estimate_topic") != g("truth_topic"):
            self.create_subscription(LocalizationData, g("estimate_topic"), self.on_est, BE)
        self.same_topic = g("estimate_topic") == g("truth_topic")
        self.create_subscription(Path, "/a2b/path", self.on_path, 1)
        self.create_subscription(String, "/a2b/status", self.on_status, 10)
        self.create_subscription(Float32MultiArray, "/pwm_duty", self.on_pwm, 10)
        if g("points_topic"):
            self.create_subscription(PointCloud2, g("points_topic"), self.on_points, BE)
        self.pub_enable = self.create_publisher(Bool, "/a2b/enable", 10)
        self.pub_goal = self.create_publisher(PoseStamped, "/goal_pose", 10)

    # ---- ROS callbacks ---------------------------------------------------------
    def on_truth(self, m):
        with self.lock:
            self.truth = (m.x, m.y, m.yaw, m.speed)
            self._trail(self.truth_trail, m.x, m.y)

    def on_est(self, m):
        if not m.is_valid:
            return
        with self.lock:
            self.est = (m.x, m.y, m.yaw, m.speed)
            self._trail(self.est_trail, m.x, m.y)

    @staticmethod
    def _trail(trail, x, y):
        if not trail or math.hypot(trail[-1][0] - x, trail[-1][1] - y) > 0.05:
            trail.append((round(x, 3), round(y, 3)))

    def on_path(self, msg):
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        with self.lock:
            self.route = [(round(x, 3), round(y, 3)) for x, y in pts[::5]] + ([pts[-1]] if pts else [])
            if pts:
                self.goal = pts[-1]

    def on_status(self, msg):
        s = msg.data
        state = s.split(" ", 1)[0]
        info = {}
        for key in ("remaining", "v_ref", "cte"):
            mm = re.search(rf"'{key}': (-?[0-9.]+)", s)
            if mm:
                info[key] = float(mm.group(1))
        mm = re.search(r"'reason': '([^']*)'", s)
        with self.lock:
            self.status, self.info = state, info
            self.reason = mm.group(1) if mm else ""

    def on_pwm(self, msg):
        if len(msg.data) >= 3:
            with self.lock:
                self.pwm = [round(float(v), 2) for v in msg.data[:3]]

    def on_points(self, msg):
        try:
            offs = {f.name: f.offset for f in msg.fields}
            n = msg.width * msg.height
            if n == 0 or not {"x", "y", "z"} <= offs.keys():
                return
            buf = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(n, msg.point_step)
            xyz = np.stack([buf[:, offs[k]:offs[k] + 4].copy().view(np.float32)[:, 0] for k in "xyz"], 1)
            xyz = xyz[np.isfinite(xyz).all(1)]
            xyz = xyz[xyz[:, 2] > -0.2]                     # drop the floor for a clear picture
            if len(xyz) > self.max_pts:
                xyz = xyz[np.linspace(0, len(xyz) - 1, self.max_pts).astype(int)]
            with self.lock:
                self.scan = np.round(xyz[:, :2], 2).tolist()
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"point cloud decode failed: {e}")

    # ---- web state -------------------------------------------------------------
    def snapshot(self):
        with self.lock:
            truth = self.truth
            est = self.est if not self.same_topic else None
            # draw the scan around the lidar-based pose (shows how well the estimate fits the room)
            base = est or truth
            scan_world = []
            if base and self.scan:
                x, y, yaw = base[0], base[1], math.radians(base[2])
                c, s = math.cos(yaw), math.sin(yaw)
                lx, ly = x + c * self.off, y + s * self.off
                scan_world = [(round(lx + c * px - s * py, 2), round(ly + s * px + c * py, 2)) for px, py in self.scan]
            err = None
            if truth and est:
                err = math.hypot(truth[0] - est[0], truth[1] - est[1])
            return {
                "t": round(time.monotonic() - self.t0, 1),
                "truth": truth, "est": est, "error": err,
                "truth_trail": list(self.truth_trail), "est_trail": list(self.est_trail),
                "route": self.route, "goal": self.goal, "scan": scan_world,
                "status": self.status, "reason": self.reason, "info": self.info, "pwm": self.pwm,
                "room": self.room,
            }

    def start(self):
        self.pub_enable.publish(Bool(data=True))

    def stop(self):
        self.pub_enable.publish(Bool(data=False))

    def set_goal(self, x, y):
        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = "odom"
        ps.pose.position.x, ps.pose.position.y = float(x), float(y)
        ps.pose.orientation.w = 1.0
        self.pub_goal.publish(ps)
        with self.lock:
            self.goal = (float(x), float(y))

    def clear_trails(self):
        with self.lock:
            self.truth_trail.clear()
            self.est_trail.clear()


def make_app(node):
    app = Flask(__name__)

    @app.get("/")
    def index():
        return Response(PAGE, mimetype="text/html")

    @app.get("/state")
    def state():
        return jsonify(node.snapshot())

    @app.post("/cmd/<name>")
    def cmd(name):
        if name == "start":
            node.start()
        elif name == "stop":
            node.stop()
        elif name == "clear":
            node.clear_trails()
        elif name == "goal":
            d = request.get_json(force=True)
            node.set_goal(d["x"], d["y"])
        else:
            return jsonify(ok=False), 404
        return jsonify(ok=True)

    return app


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>A→B Live Sim</title>
<style>
:root{--bg:#12161b;--panel:#1b2128;--line:#2e3742;--ink:#e6eaee;--muted:#93a0ad;--truth:#e6eaee;
--est:#ffb319;--route:#5aa9e6;--goal:#ff6b6b;--scan:#4fd1a5;--ok:#6fd39c;--stop:#ff8a80}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif;
display:grid;grid-template-columns:1fr 300px;height:100vh}
#map{width:100%;height:100%;display:block;cursor:crosshair}
aside{background:var(--panel);border-left:1px solid var(--line);padding:16px;display:flex;flex-direction:column;gap:14px;overflow:auto}
h1{font-size:18px;margin:0}.muted{color:var(--muted);font-size:12px}
.state{font:600 15px/1 monospace;padding:8px 10px;border-radius:6px;background:var(--line);text-align:center}
.state.DRIVING{background:#3a2e12;color:var(--est)}.state.ARRIVED{background:#173327;color:var(--ok)}
.state.STOPPED{background:#3a1d1b;color:var(--stop)}
dl{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;margin:0;font-variant-numeric:tabular-nums}
dt{color:var(--muted)}dd{margin:0;text-align:right;font-family:monospace}
.btns{display:grid;grid-template-columns:1fr 1fr;gap:8px}
button{font:600 14px system-ui;padding:10px;border-radius:6px;border:1px solid var(--line);background:#232a33;color:var(--ink);cursor:pointer}
button.go{background:#1f5a3a;border-color:#2d7a50}button.halt{background:#6a2420;border-color:#8a332d}
button:focus-visible{outline:2px solid var(--est);outline-offset:2px}
.bar{height:10px;background:#232a33;border-radius:5px;position:relative;overflow:hidden}
.bar i{position:absolute;top:0;bottom:0;background:var(--est)}.bar b{position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--muted)}
.legend span{display:inline-flex;align-items:center;gap:6px;margin-right:12px;font-size:12px;color:var(--muted)}
.legend i{width:14px;height:3px;display:inline-block}
@media (max-width:760px){body{grid-template-columns:1fr;grid-template-rows:60vh auto;height:auto}}
</style></head><body>
<canvas id="map"></canvas>
<aside>
  <div><h1>A→B live simulation</h1><div class="muted">Click the map to set B, then Start.</div></div>
  <div id="state" class="state">IDLE</div>
  <div id="reason" class="muted"></div>
  <div class="btns"><button class="go" id="start">Start</button><button class="halt" id="stop">Stop</button>
  <button id="clear" style="grid-column:span 2">Clear trails</button></div>
  <dl>
    <dt>Time</dt><dd id="t">-</dd>
    <dt>Speed</dt><dd id="v">-</dd>
    <dt>Target speed</dt><dd id="vref">-</dd>
    <dt>Distance to B</dt><dd id="rem">-</dd>
    <dt>Off route</dt><dd id="cte">-</dd>
    <dt>Lidar error</dt><dd id="err">-</dd>
    <dt>True pose</dt><dd id="tp">-</dd>
    <dt>Lidar pose</dt><dd id="ep">-</dd>
  </dl>
  <div><div class="muted">PWM1 steering <span id="p1"></span>%</div><div class="bar"><b></b><i id="b1"></i></div></div>
  <div><div class="muted">PWM3 drive <span id="p3"></span>%</div><div class="bar"><b></b><i id="b3"></i></div></div>
  <div class="legend"><span><i style="background:var(--truth)"></i>true</span><span><i style="background:var(--est)"></i>lidar</span>
  <span><i style="background:var(--route)"></i>route</span><span><i style="background:var(--scan)"></i>scan</span></div>
</aside>
<script>
const cv=document.getElementById('map'),ctx=cv.getContext('2d');let S=null,view=null;
function css(n){return getComputedStyle(document.documentElement).getPropertyValue(n).trim()}
function resize(){const r=cv.getBoundingClientRect(),d=devicePixelRatio||1;cv.width=r.width*d;cv.height=r.height*d;ctx.setTransform(d,0,0,d,0,0)}
addEventListener('resize',()=>{resize();view=null});resize();
function bounds(){let xs=[],ys=[];const add=(x,y)=>{xs.push(x);ys.push(y)};
 if(S.room){S.room.walls.forEach(w=>{add(w[0],w[1]);add(w[2],w[3])})}
 else{(S.truth_trail||[]).forEach(p=>add(p[0],p[1]));(S.route||[]).forEach(p=>add(p[0],p[1]));
  if(S.truth)add(S.truth[0],S.truth[1]);if(S.goal)add(S.goal[0],S.goal[1])}
 if(!xs.length)return null;let x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
 const pad=Math.max(2,0.08*Math.max(x1-x0,y1-y0));return[x0-pad,y0-pad,x1+pad,y1+pad]}
function fit(){const b=bounds();if(!b)return;const r=cv.getBoundingClientRect();
 const s=Math.min(r.width/(b[2]-b[0]),r.height/(b[3]-b[1]));view={s,cx:(b[0]+b[2])/2,cy:(b[1]+b[3])/2,w:r.width,h:r.height}}
function P(x,y){return[view.w/2+(x-view.cx)*view.s,view.h/2-(y-view.cy)*view.s]}
function W(px,py){return[(px-view.w/2)/view.s+view.cx,-(py-view.h/2)/view.s+view.cy]}
function inView(x,y){const[p,q]=P(x,y);return p>20&&q>20&&p<view.w-20&&q<view.h-20}
function line(pts,color,w,dash){if(pts.length<2)return;ctx.beginPath();ctx.setLineDash(dash||[]);
 pts.forEach((p,i)=>{const[a,b]=P(p[0],p[1]);i?ctx.lineTo(a,b):ctx.moveTo(a,b)});ctx.strokeStyle=color;ctx.lineWidth=w;ctx.stroke();ctx.setLineDash([])}
function truck(pose,color,fill){const[x,y,yawd]=pose,yaw=yawd*Math.PI/180,L=Math.max(0.55,10/view.s),Wd=L*0.6;
 const c=Math.cos(yaw),s=Math.sin(yaw),pts=[[-0.15*L,-Wd/2],[0.85*L,-Wd/2],[1.05*L,0],[0.85*L,Wd/2],[-0.15*L,Wd/2]];
 ctx.beginPath();pts.forEach((p,i)=>{const[a,b]=P(x+c*p[0]-s*p[1],y+s*p[0]+c*p[1]);i?ctx.lineTo(a,b):ctx.moveTo(a,b)});ctx.closePath();
 if(fill){ctx.fillStyle=color;ctx.fill()}else{ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke()}}
function draw(){const r=cv.getBoundingClientRect();ctx.fillStyle=css('--bg');ctx.fillRect(0,0,r.width,r.height);if(!S)return;
 if(!view||(S.truth&&!inView(S.truth[0],S.truth[1]))||(S.goal&&!inView(S.goal[0],S.goal[1])))fit();if(!view)return;
 // 1 m grid
 const step=view.s<8?5:1;const[x0,y1]=W(0,0),[x1,y0]=W(r.width,r.height);ctx.strokeStyle='#1d242c';ctx.lineWidth=1;ctx.beginPath();
 for(let x=Math.floor(x0/step)*step;x<x1;x+=step){const[a]=P(x,0);ctx.moveTo(a,0);ctx.lineTo(a,r.height)}
 for(let y=Math.floor(y0/step)*step;y<y1;y+=step){const[,b]=P(0,y);ctx.moveTo(0,b);ctx.lineTo(r.width,b)}ctx.stroke();
 if(S.room){ctx.strokeStyle='#5b6672';ctx.lineWidth=3;S.room.walls.forEach(w=>{const[a,b]=P(w[0],w[1]),[c,d]=P(w[2],w[3]);ctx.beginPath();ctx.moveTo(a,b);ctx.lineTo(c,d);ctx.stroke()});
  ctx.fillStyle='#2a323c';S.room.boxes.forEach(b=>{const[a,c]=P(b[0],b[3]),[d,e]=P(b[2],b[1]);ctx.fillRect(a,c,d-a,e-c)});
  S.room.pillars.forEach(p=>{const[a,b]=P(p[0],p[1]);ctx.beginPath();ctx.arc(a,b,Math.max(3,p[2]*view.s),0,7);ctx.fill()})}
 ctx.fillStyle=css('--scan');(S.scan||[]).forEach(p=>{const[a,b]=P(p[0],p[1]);ctx.fillRect(a-1,b-1,2,2)});
 line(S.route||[],css('--route'),2,[6,5]);line(S.truth_trail||[],css('--truth'),2);line(S.est_trail||[],css('--est'),2,[4,3]);
 if(S.goal){const[a,b]=P(S.goal[0],S.goal[1]);ctx.strokeStyle=css('--goal');ctx.lineWidth=2;ctx.beginPath();ctx.arc(a,b,9,0,7);ctx.stroke();
  ctx.fillStyle=css('--goal');ctx.font='600 13px system-ui';ctx.fillText('B',a+12,b-10)}
 if(S.route&&S.route.length){const[a,b]=P(S.route[0][0],S.route[0][1]);ctx.fillStyle=css('--route');ctx.fillText('A',a+10,b-10)}
 if(S.est)truck(S.est,css('--est'),false);if(S.truth)truck(S.truth,css('--truth'),true)}
const $=id=>document.getElementById(id);
const f=(v,d=2,u='')=>v==null?'-':(+v).toFixed(d)+u;
function panel(){if(!S)return;const st=document.getElementById('state');st.textContent=S.status;st.className='state '+S.status;
 document.getElementById('reason').textContent=S.reason||'';const pose=S.truth||S.est,i=S.info||{};
 $('t').textContent=f(S.t,1,' s');$('v').textContent=pose?f(pose[3],2,' m/s'):'-';$('vref').textContent=f(i.v_ref,2,' m/s');
 $('rem').textContent=f(i.remaining,2,' m');$('cte').textContent=f(i.cte,2,' m');$('err').textContent=f(S.error,2,' m');
 $('tp').textContent=S.truth?`${f(S.truth[0],1)}, ${f(S.truth[1],1)}, ${f(S.truth[2],0)}°`:'-';
 $('ep').textContent=S.est?`${f(S.est[0],1)}, ${f(S.est[1],1)}, ${f(S.est[2],0)}°`:'-';
 [[1,0],[3,2]].forEach(([n,k])=>{const d=S.pwm[k];document.getElementById('p'+n).textContent=d.toFixed(2);
  const b=document.getElementById('b'+n),lo=Math.min(d,7.5),hi=Math.max(d,7.5);b.style.left=((lo-5)/5*100)+'%';b.style.width=((hi-lo)/5*100)+'%'})}
async function poll(){try{S=await (await fetch('/state')).json();panel();draw()}catch(e){}setTimeout(poll,100)}poll();
const post=(p,b)=>fetch('/cmd/'+p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});
$('start').onclick=()=>post('start');$('stop').onclick=()=>post('stop');$('clear').onclick=()=>post('clear');
cv.addEventListener('click',e=>{if(!view)return;const r=cv.getBoundingClientRect(),[x,y]=W(e.clientX-r.left,e.clientY-r.top);post('goal',{x,y})});
</script></body></html>"""


def main():
    rclpy.init()
    node = SimViewer()
    if Flask is None:
        node.get_logger().error("Flask is not installed: sudo apt install python3-flask")
        node.destroy_node()
        rclpy.shutdown()
        return
    app = make_app(node)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)   # no log line per page update
    web = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=node.port, threaded=True,
                                                  use_reloader=False), daemon=True)
    web.start()
    node.get_logger().info(f"live view: http://localhost:{node.port}")
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
