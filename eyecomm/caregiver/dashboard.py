"""Tkinter caregiver dashboard: board customisation and request alerts."""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
import requests
import threading

from eyecomm.services.settings import load_settings
from eyecomm.server.store import DEFAULT_TILES

class CaregiverDashboard:
    def __init__(self, root):
        self.root = root
        self.settings = load_settings()
        self.base_url = self.settings["server_url"]
        self.root.title("EyeComm - Caregiver Dashboard")
        self.root.geometry("1000x720")
        self.root.minsize(820, 600)
        self.seen_ids = set()
        self.rows = {}
        self.tile_vars = []
        self._build()
        self.load_board()
        self.refresh_requests()

    def _build(self):
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="EYECOMM CAREGIVER", font=("Segoe UI", 18, "bold")).pack(anchor="w")
        self.status = ttk.Label(outer, text=f"Server: {self.base_url}")
        self.status.pack(anchor="w", pady=(2, 10))

        top = ttk.Panedwindow(outer, orient="horizontal")
        top.pack(fill="both", expand=True)

        board_frame = ttk.LabelFrame(top, text="Patient communication board", padding=10)
        request_frame = ttk.LabelFrame(top, text="Patient requests", padding=10)
        top.add(board_frame, weight=1)
        top.add(request_frame, weight=2)

        ttk.Label(board_frame, text="Edit the label and spoken phrase. Action type stays fixed.").pack(anchor="w")
        headers = ttk.Frame(board_frame)
        headers.pack(fill="x", pady=(8, 2))
        ttk.Label(headers, text="Button label", width=18).pack(side="left")
        ttk.Label(headers, text="Spoken phrase", width=25).pack(side="left")

        for tile in DEFAULT_TILES:
            line = ttk.Frame(board_frame)
            line.pack(fill="x", pady=3)
            label_var = tk.StringVar(value=tile["label"])
            phrase_var = tk.StringVar(value=tile["phrase"])
            ttk.Label(line, text=tile["id"], width=8).pack(side="left")
            ttk.Entry(line, textvariable=label_var, width=18).pack(side="left", padx=3)
            ttk.Entry(line, textvariable=phrase_var, width=25).pack(side="left", padx=3)
            self.tile_vars.append((tile["id"], tile["action"], label_var, phrase_var))

        ttk.Button(board_frame, text="Save patient board", command=self.save_board).pack(anchor="e", pady=10)

        cols = ("patient", "message", "priority", "status", "created")
        self.tree = ttk.Treeview(request_frame, columns=cols, show="headings", height=18)
        for col, width in [
            ("patient", 100), ("message", 230), ("priority", 80),
            ("status", 110), ("created", 150)
        ]:
            self.tree.heading(col, text=col.title())
            self.tree.column(col, width=width, anchor="w")
        self.tree.pack(fill="both", expand=True)

        actions = ttk.Frame(request_frame)
        actions.pack(fill="x", pady=8)
        ttk.Button(actions, text="Acknowledge", command=self.acknowledge).pack(side="left", padx=3)
        ttk.Button(actions, text="Mark resolved", command=self.resolve).pack(side="left", padx=3)
        ttk.Button(actions, text="Refresh", command=self.refresh_now).pack(side="right")

        ttk.Label(outer, text="Urgent requests produce an audible alert. Keep this dashboard running and notifications enabled.",
                  foreground="#8b3d00").pack(anchor="w", pady=(8, 0))

    def _get(self, path):
        return requests.get(self.base_url + path, timeout=2)

    def load_board(self):
        try:
            data = self._get("/api/board").json()
            by_id = {t["id"]: t for t in data["tiles"]}
            for tile_id, action, label_var, phrase_var in self.tile_vars:
                tile = by_id.get(tile_id)
                if tile:
                    label_var.set(tile["label"])
                    phrase_var.set(tile.get("phrase", ""))
            self.status.config(text=f"Connected to {self.base_url}")
        except Exception as exc:
            self.status.config(text=f"Server unavailable: {exc}")

    def save_board(self):
        tiles = []
        for tile_id, action, label_var, phrase_var in self.tile_vars:
            tiles.append({
                "id": tile_id,
                "action": action,
                "label": label_var.get().strip(),
                "phrase": phrase_var.get().strip()
            })
        try:
            response = requests.put(
                self.base_url + "/api/board",
                json={"tiles": tiles},
                timeout=3
            )
            response.raise_for_status()
            messagebox.showinfo("EyeComm", "Patient communication board saved.")
        except Exception as exc:
            messagebox.showerror("EyeComm", f"Could not save board:\n{exc}")

    def _sound_alert(self):
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except Exception:
            try:
                self.root.bell()
            except Exception:
                pass

    def refresh_requests(self):
        try:
            response = self._get("/api/requests")
            response.raise_for_status()
            items = response.json().get("requests", [])
            for item in items:
                rid = item["request_id"]
                values = (
                    item["patient_id"], item["message"], item["priority"].upper(),
                    item["status"], item["created_at"][:19].replace("T", " ")
                )
                if rid in self.rows:
                    self.tree.item(self.rows[rid], values=values)
                else:
                    row_id = self.tree.insert("", "end", values=values)
                    self.rows[rid] = row_id
                if rid not in self.seen_ids:
                    self.seen_ids.add(rid)
                    if item["priority"] == "urgent" and item["status"] == "PENDING":
                        self._sound_alert()
            self.status.config(text=f"Connected to {self.base_url} · {len(items)} recent requests")
        except Exception as exc:
            self.status.config(text=f"Waiting for EyeComm server: {exc}")
        self.root.after(1800, self.refresh_requests)

    def _selected_request_id(self):
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("EyeComm", "Select a request first.")
            return None
        selected_item = selection[0]
        for request_id, row_id in self.rows.items():
            if row_id == selected_item:
                return request_id
        return None

    def acknowledge(self):
        request_id = self._selected_request_id()
        if not request_id:
            return
        try:
            requests.post(
                f"{self.base_url}/api/requests/{request_id}/ack",
                json={"caregiver": "Caregiver"},
                timeout=3
            ).raise_for_status()
            self.refresh_now()
        except Exception as exc:
            messagebox.showerror("EyeComm", str(exc))

    def resolve(self):
        request_id = self._selected_request_id()
        if not request_id:
            return
        try:
            requests.post(
                f"{self.base_url}/api/requests/{request_id}/resolve",
                timeout=3
            ).raise_for_status()
            self.refresh_now()
        except Exception as exc:
            messagebox.showerror("EyeComm", str(exc))

    def refresh_now(self):
        # Refresh immediately without scheduling a second polling chain.
        try:
            items = self._get("/api/requests").json().get("requests", [])
            for item in items:
                rid = item["request_id"]
                values = (item["patient_id"], item["message"], item["priority"].upper(),
                          item["status"], item["created_at"][:19].replace("T", " "))
                if rid in self.rows:
                    self.tree.item(self.rows[rid], values=values)
        except Exception:
            pass

def main():
    root = tk.Tk()
    CaregiverDashboard(root)
    root.mainloop()

if __name__ == "__main__":
    main()
