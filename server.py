#!/usr/bin/env python3
"""
回响 — 文件接收服务器
运行在老板的Mac上，接收用户上传的聊天记录

启动: python3 server.py
然后 Cloudflare Tunnel 暴露到公网
"""
import os, json, hashlib, uuid, threading, time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs
from datetime import datetime

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
ORDERS_FILE = os.path.join(os.path.dirname(__file__), "orders.json")
PORT = 8888

os.makedirs(UPLOAD_DIR, exist_ok=True)

class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
    
    def _json(self, data, code=200):
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())
    
    def do_OPTIONS(self):
        self._json({"ok": True})
    
    def do_GET(self):
        if self.path == "/health":
            self._json({"status": "ok", "time": datetime.now().isoformat()})
        elif self.path.startswith("/orders"):
            self._list_orders()
        else:
            self._json({"error": "not found"}, 404)
    
    def do_POST(self):
        if self.path == "/upload":
            self._handle_upload()
        elif self.path == "/order":
            self._handle_order()
        else:
            self._json({"error": "not found"}, 404)
    
    def _handle_upload(self):
        """接收聊天记录文件"""
        content_type = self.headers.get("Content-Type", "")
        
        if "multipart/form-data" not in content_type:
            self._json({"error": "需要multipart上传"}, 400)
            return
        
        # 解析multipart（简易版）
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        
        boundary = content_type.split("boundary=")[1].encode()
        parts = body.split(b"--" + boundary)
        
        saved_files = []
        for part in parts:
            if b"Content-Disposition" not in part:
                continue
            
            # 提取文件名
            header_match = part.split(b"\r\n\r\n", 1)
            if len(header_match) < 2:
                continue
            header, data = header_match
            data = data.rsplit(b"\r\n", 1)[0]
            
            if b'filename="' in header:
                filename = header.split(b'filename="')[1].split(b'"')[0].decode()
                # 安全的文件名
                safe_name = f"{uuid.uuid4().hex[:8]}_{filename}"
                filepath = os.path.join(UPLOAD_DIR, safe_name)
                with open(filepath, "wb") as f:
                    f.write(data)
                saved_files.append({
                    "original": filename,
                    "saved": safe_name,
                    "size": len(data)
                })
        
        if saved_files:
            self._json({
                "success": True,
                "files": saved_files,
                "message": f"收到 {len(saved_files)} 个文件"
            })
        else:
            self._json({"error": "没有收到文件"}, 400)
    
    def _handle_order(self):
        """创建订单（收到文件后）"""
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))
        
        order_id = uuid.uuid4().hex[:12]
        order = {
            "id": order_id,
            "name": body.get("name", ""),
            "email": body.get("email", ""),
            "files": body.get("files", []),
            "status": "received",
            "created": datetime.now().isoformat(),
            "paid": False
        }
        
        # 保存订单
        orders = []
        if os.path.exists(ORDERS_FILE):
            with open(ORDERS_FILE, "r") as f:
                orders = json.load(f)
        orders.append(order)
        with open(ORDERS_FILE, "w") as f:
            json.dump(orders, f, ensure_ascii=False, indent=2)
        
        self._json({
            "success": True,
            "order_id": order_id,
            "message": f"订单 {order_id} 已创建，24小时内完成训练"
        })
    
    def _list_orders(self):
        if os.path.exists(ORDERS_FILE):
            with open(ORDERS_FILE, "r") as f:
                orders = json.load(f)
        else:
            orders = []
        self._json({"orders": orders})
    
    def log_message(self, format, *args):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {args[0]}")


def main():
    print(f"""
╔══════════════════════════════════════════╗
║         回响 · 文件接收服务器             ║
║         端口: {PORT}                       ║
║         上传目录: {UPLOAD_DIR}            ║
╚══════════════════════════════════════════╝
    """)
    
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止")


if __name__ == "__main__":
    main()
