import os
import platform
import re
import socket
import subprocess
import threading
import time
import tkinter as tk
from copy import deepcopy
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

import yaml
from chardet import UniversalDetector

# default data
DEFAULT_CONFIG = {
    "ffmpeg_framerate": 6,
    "image": {"dpi": 100, "width": 1920, "heigth": 1080},
    "input_file": "",
    "steps": [],
}

DEFAULT_STEP = {
    "name": "New Step",
    "sim_command": ".ac dec 100 20 20k",
    "tracestoplot": [],
    "parameters": {},
}
LTSPICEBATCH = "LTSpiceBatch.py"
PLATFORM_SYSTEM = platform.system()

param_names: str = ""

def get_string(s: str):
    if PLATFORM_SYSTEM == "Windows":
        return s
    else:
        return "".join([c for c in s if ord(c) < 127]).strip()


class TCPClient:
    def __init__(self, host="localhost", port=5000):
        self.host = host
        self.port = port
        self.client_socket = None

    def connect(self):
        """Établit la connexion avec le serveur."""
        attempt = 0
        while attempt <= 20:
            try:
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket.connect((self.host, self.port))
                # print(f"[CLIENT] Connecté au serveur {self.host}:{self.port}")
            except (ConnectionRefusedError, OSError):
                # print(f"[CLIENT] Try {attempt}/20")
                self.client_socket.close()
                time.sleep(1)
            finally:
                attempt += 1

    def send(self, message: str) -> str:
        """Envoie un message et retourne la réponse du serveur."""
        self.client_socket.sendall(message.encode("utf-8"))
        response = self.client_socket.recv(1024).decode("utf-8")
        # print(f"[CLIENT] Réponse du serveur : {response}")
        return response

    def disconnect(self):
        """Ferme la connexion."""
        if self.client_socket:
            self.client_socket.close()
        # print("[CLIENT] Déconnecté.")


# main window
class YamlGeneratorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LTSpiceBatch - Config and Run")
        self.geometry("1100x800")
        self.resizable(True, True)
        self.configure(bg="#2b2b2b")

        self.config_data = deepcopy(DEFAULT_CONFIG)
        self.steps = []  # liste de StepFrame

        self._build_styles()
        self._build_ui()

    
    # ttk styles    
    def _build_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        BG = "#2b2b2b"
        FG = "#dcdcdc"
        ENTRY_BG = "#3c3f41"
        ACCENT = "#4a90d9"
        BTN_BG = "#4a90d9"
        BTN_FG = "#ffffff"
        FRAME_BG = "#313335"

        style.configure("TFrame", background=BG)
        style.configure(
            "Card.TFrame", background=FRAME_BG, relief="flat", borderwidth=1
        )
        style.configure("TLabel", background=BG, foreground=FG, font=("Segoe UI", 9))
        style.configure(
            "Title.TLabel",
            background=BG,
            foreground=ACCENT,
            font=("Segoe UI", 12, "bold"),
        )
        style.configure(
            "CardTitle.TLabel",
            background=FRAME_BG,
            foreground=ACCENT,
            font=("Segoe UI", 10, "bold"),
        )
        style.configure(
            "Card.TLabel", background=FRAME_BG, foreground=FG, font=("Segoe UI", 9)
        )
        style.configure(
            "TEntry",
            fieldbackground=ENTRY_BG,
            foreground=FG,
            insertcolor=FG,
            borderwidth=1,
        )
        style.configure(
            "TButton",
            background=BTN_BG,
            foreground=BTN_FG,
            font=("Segoe UI", 9, "bold"),
            borderwidth=0,
            focusthickness=0,
        )
        style.map("TButton", background=[("active", "#357abd"), ("pressed", "#2a6099")])
        style.configure(
            "Danger.TButton",
            background="#c0392b",
            foreground="white",
            font=("Segoe UI", 9, "bold"),
        )
        style.map("Danger.TButton", background=[("active", "#e74c3c")])
        style.configure(
            "Success.TButton",
            background="#27ae60",
            foreground="white",
            font=("Segoe UI", 9, "bold"),
        )
        style.map("Success.TButton", background=[("active", "#2ecc71")])
        style.configure(
            "TLabelframe",
            background=FRAME_BG,
            foreground=ACCENT,
            font=("Segoe UI", 9, "bold"),
        )
        style.configure(
            "TLabelframe.Label",
            background=FRAME_BG,
            foreground=ACCENT,
            font=("Segoe UI", 9, "bold"),
        )
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background="#3c3f41",
            foreground=FG,
            padding=[10, 4],
            font=("Segoe UI", 9),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", FRAME_BG)],
            foreground=[("selected", ACCENT)],
        )
        style.configure(
            "TScrollbar", background=ENTRY_BG, troughcolor=BG, arrowcolor=FG
        )
        style.configure(
            "TSpinbox", fieldbackground=ENTRY_BG, foreground=FG, insertcolor=FG
        )

    def _load_config(self):
        """load last working params"""
        data = {}
        config_file = Path(__file__).parent.joinpath(".config.yml")
        if config_file.exists():
            self.status_var.set("loading last working paths")
            with open(config_file, "r", encoding="utf-8") as f:
                data: dict = yaml.safe_load(f)
                if data is None:
                    return
                if data.get("ffmpeg_bin", "") != "":
                    self.var_ffmpeg_bin.set(data.get("ffmpeg_bin"))
                if PLATFORM_SYSTEM == "Linux":
                    if data.get("wine_executable", "") != "":
                        self.var_wine_executable.set(data.get("wine_executable"))
                    if data.get("wine_folder", "") != "":
                        self.var_wine_folder.set(data.get("wine_folder"))
        self.status_var.set(get_string("✅ Loaded: last working paths"))

    def _save_config(self):
        """save last working params"""
        data = {}
        config_file = Path(__file__).parent.joinpath(".config.yml")
        self.status_var.set("saving last working paths")
        if getattr(self, "var_ffmpeg_bin", "") != "":
            data["ffmpeg_bin"] = self.var_ffmpeg_bin.get()
        if PLATFORM_SYSTEM == "Linux":
            if getattr(self, "var_wine_executable", "") != "":
                data["wine_executable"] = self.var_wine_executable.get()
            if getattr(self, "var_wine_folder", "") != "":
                data["wine_folder"] = self.var_wine_folder.get()
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(self._to_yaml_string(data=data))
        self.status_var.set(get_string("✅ Saved: last working paths"))

    
    # build interface
    
    def _build_ui(self):
        # ---- title ----
        header = ttk.Frame(self)
        header.pack(fill="x", padx=10, pady=(10, 0))
        ttk.Label(
            header,
            text=get_string("⚙  LTSpiceBatch - Config and Run"),
            style="Title.TLabel",
        ).pack(side="left")

        # ---- main notebook ----
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # tab 1 : main configuration
        self.tab_global = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_global, text=get_string("  🌐 Global Config  "))
        self._build_global_tab()

        # tab 2 : steps
        self.tab_steps = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_steps, text=get_string("  📋 Steps  "))
        self._build_steps_tab()

        # tab 3 : yaml preview
        self.tab_preview = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_preview, text=get_string("  👁  Preview YAML  "))
        self._build_preview_tab()

        # tab 4 : Exec it
        self.tab_exec = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_exec, text=get_string("  ▶️  Save and Run  "))
        self._build_exec_tab()

        # ---- status bar ----
        self._build_status_bar()
        self._load_config()

    
    # global tab
    
    def _build_global_tab(self):
        self.global_canvas = tk.Canvas(
            self.tab_global, bg="#2b2b2b", highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(
            self.tab_global, orient="vertical", command=self.global_canvas.yview
        )
        self.global_frame = ttk.Frame(self.global_canvas)

        self.global_frame.bind(
            "<Configure>",
            lambda e: self.global_canvas.configure(
                scrollregion=self.global_canvas.bbox("all")
            ),
        )

        self.global_canvas.create_window((0, 0), window=self.global_frame, anchor="nw")
        self.global_canvas.configure(yscrollcommand=scrollbar.set)

        self.global_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        f = self.global_frame

        # ---- ffmpeg ----
        lf1 = ttk.LabelFrame(f, text=get_string(" 🎬 FFmpeg Settings "), padding=10)
        lf1.pack(fill="x", padx=15, pady=10)

        self._labeled_entry(
            lf1,
            "ffmpeg_bin :",
            "ffmpeg_bin",
            "/opt/python-uv/uv-linux/ffmpeg",
            width=60,
            browse_file=True,
        )
        self._labeled_entry(
            lf1, "ffmpeg_framerate :", "ffmpeg_framerate", "6", width=10
        )

        # ---- image ----
        lf2 = ttk.LabelFrame(f, text=get_string(" 🖼  Image Settings "), padding=10)
        lf2.pack(fill="x", padx=15, pady=10)

        self._labeled_entry(lf2, "dpi :", "image_dpi", "100", width=10)
        self._labeled_entry(lf2, "width (px) :", "image_width", "1920", width=10)
        self._labeled_entry(lf2, "height (px) :", "image_heigth", "1080", width=10)

        # ---- simulation ----
        lf3 = ttk.LabelFrame(f, text=get_string(" ⚡ Simulation Settings "), padding=10)
        lf3.pack(fill="x", padx=15, pady=10)

        self._labeled_entry(
            lf3,
            "input_file :",
            "input_file",
            "",
            width=60,
            browse_file=True,
            filetypes=[("ASC files", "*.asc"), ("All files", "*.*")],
        )
        self._labeled_entry(lf3, "parallel_sim :", "parallel_sim", "", width=10)
        self._labeled_entry(lf3, "parallel_plot :", "parallel_plot", "", width=10)
        self._labeled_entry(lf3, "runner_timeout :", "runner_timeout", "", width=10)
        # ---- opt ----
        lf4 = ttk.LabelFrame(f, text=get_string(" 📁 Optional "), padding=10)
        lf4.pack(fill="x", padx=15, pady=10)
        self._labeled_entry(
            lf4, "temp_folder :", "temp_folder", "", width=60, browse_dir=True
        )
        self._labeled_checkbox(
            lf4,
            "only encode :\nencode from previously generated images",
            "encode_only",
            False,
            width=50,
        )
        self._labeled_checkbox(
            lf4, "keep images :\nkeep generated images", "keep_images", False, width=50
        )
        self._labeled_checkbox(lf4, "cleanup temp dir:", "cleanup", False, width=50)
        self._labeled_checkbox(
            lf4, "show frequency domains:", "show_freq_domains", False, width=50
        )
        self._labeled_checkbox(
            lf4,
            "reset:\nstart over, even if the job could be resumed",
            "reset",
            False,
            width=50,
        )
        if PLATFORM_SYSTEM == "Linux":
            self._labeled_entry(
                lf4,
                "wine executable :\nto set custom location or appimage path",
                "wine_executable",
                "",
                width=10,
                browse_file=True,
                filetypes=[("All files", "*.*")],
            )
            self._labeled_entry(
                lf4,
                "wine folder :\nto set .wine directory",
                "wine_folder",
                "",
                width=10,
                browse_dir=True,
            )

        # mouse wheel bind
        self.global_canvas.bind("<MouseWheel>", self._on_mousewheel_global)

    def _labeled_entry(
        self,
        parent,
        label,
        key,
        default,
        width=30,
        browse_file=False,
        browse_dir=False,
        filetypes=None,
    ):
        """line with label + entry + opt brows button"""
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, width=20, anchor="w").pack(side="left")

        var = tk.StringVar(value=default)
        setattr(self, f"var_{key}", var)

        def trace_add_callback(var, index, mode):
            if label.startswith("input_file"):
                global param_names
                param_names = self._get_param_names()
                if hasattr(self, "loaded_data"):
                    self._populate_from_data(self.loaded_data)

        var.trace_add("write", trace_add_callback)

        entry = ttk.Entry(row, textvariable=var, width=width)
        entry.pack(side="left", padx=(0, 5))

        if browse_file:

            def _browse(v=var, ft=filetypes):
                path = filedialog.askopenfilename(
                    filetypes=ft or [("All files", "*.*")],
                    initialdir=Path.home()
                )
                if path:
                    v.set(path)

            ttk.Button(
                row, text=get_string("📂 Browse"), command=_browse, width=10
            ).pack(side="left")

        if browse_dir:

            def _browse_dir(v=var):
                path = filedialog.askdirectory()
                if path:
                    v.set(path)

            ttk.Button(
                row, text=get_string("📂 Browse"), command=_browse_dir, width=10
            ).pack(side="left")

    def _labeled_checkbox(self, parent, label, key, default, width=30):
        """line with label + checkbutton"""
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, width=width, anchor="w").pack(side="left")

        var = tk.BooleanVar(value=default)
        setattr(self, f"var_{key}", var)

        entry = ttk.Checkbutton(row, variable=var, onvalue=True, offvalue=False)
        entry.pack(side="left", padx=(0, 5))

    
    # steps tab
    
    def _build_steps_tab(self):
        # add step button
        top_bar = ttk.Frame(self.tab_steps)
        top_bar.pack(fill="x", padx=10, pady=(8, 0))
        ttk.Button(
            top_bar, text=get_string("➕  Add Step"), command=self._add_step
        ).pack(side="left")
        ttk.Label(
            top_bar,
            text="Each step = one simulation run with parameter sweep",
            style="TLabel",
        ).pack(side="left", padx=15)

        # steps scrollable zone
        container = ttk.Frame(self.tab_steps)
        container.pack(fill="both", expand=True, padx=10, pady=8)

        self.steps_canvas = tk.Canvas(container, bg="#2b2b2b", highlightthickness=0)
        vsb = ttk.Scrollbar(
            container, orient="vertical", command=self.steps_canvas.yview
        )
        self.steps_canvas.configure(yscrollcommand=vsb.set)

        vsb.pack(side="right", fill="y")
        self.steps_canvas.pack(side="left", fill="both", expand=True)

        self.steps_inner = ttk.Frame(self.steps_canvas)
        self._steps_window = self.steps_canvas.create_window(
            (0, 0), window=self.steps_inner, anchor="nw"
        )

        self.steps_inner.bind("<Configure>", self._on_steps_configure)
        self.steps_canvas.bind("<Configure>", self._on_canvas_configure)

        # mouse wheel bind
        self.steps_canvas.bind("<MouseWheel>", self._on_mousewheel_steps)

    def _on_steps_configure(self, event):
        self.steps_canvas.configure(scrollregion=self.steps_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.steps_canvas.itemconfig(self._steps_window, width=event.width)

    def _on_mousewheel_global(self, event):
        self.global_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_steps(self, event):
        self.steps_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _add_step(self, data=None):
        sf = StepFrame(self.steps_inner, self, index=len(self.steps), data=data)
        sf.pack(fill="x", padx=5, pady=5)
        self.steps.append(sf)
        self._renumber_steps()

    def _remove_step(self, step_frame):
        if step_frame in self.steps:
            self.steps.remove(step_frame)
        step_frame.destroy()
        self._renumber_steps()

    def _renumber_steps(self):
        for i, sf in enumerate(self.steps):
            sf.set_index(i)

    def _move_step_up(self, step_frame):
        idx = self.steps.index(step_frame)
        if idx == 0:
            return
        self.steps[idx], self.steps[idx - 1] = self.steps[idx - 1], self.steps[idx]
        self._repack_steps()

    def _move_step_down(self, step_frame):
        idx = self.steps.index(step_frame)
        if idx == len(self.steps) - 1:
            return
        self.steps[idx], self.steps[idx + 1] = self.steps[idx + 1], self.steps[idx]
        self._repack_steps()

    def _repack_steps(self):
        for sf in self.steps:
            sf.pack_forget()
        for sf in self.steps:
            sf.pack(fill="x", padx=5, pady=5)
        self._renumber_steps()

    
    # preview tab
    
    def _build_preview_tab(self):
        bar = ttk.Frame(self.tab_preview)
        bar.pack(fill="x", padx=10, pady=(8, 0))
        ttk.Button(
            bar, text=get_string("🔄  Refresh Preview"), command=self._refresh_preview
        ).pack(side="left")
        ttk.Button(
            bar,
            text=get_string("📋  Copy to Clipboard"),
            command=self._copy_to_clipboard,
        ).pack(side="left", padx=5)

        self.preview_text = scrolledtext.ScrolledText(
            self.tab_preview,
            font=("Courier New", 10),
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="white",
            wrap="none",
            relief="flat",
        )
        self.preview_text.pack(fill="both", expand=True, padx=10, pady=8)

    def _refresh_preview(self):
        try:
            data = self._collect_data()
            yaml_str = self._to_yaml_string(data)
            self.preview_text.delete("1.0", "end")
            self.preview_text.insert("1.0", yaml_str)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _copy_to_clipboard(self):
        self._refresh_preview()
        content = self.preview_text.get("1.0", "end")
        self.clipboard_clear()
        self.clipboard_append(content)
        messagebox.showinfo("Copied", "YAML copied to clipboard!")

    
    # save and run tab
    
    def _build_exec_tab(self):
        bar = ttk.Frame(self.tab_exec)
        bar.pack(fill="x", padx=10, pady=(8, 0))
        self.exec_button = ttk.Button(
            bar, text=get_string("🔄  Save and Run"), command=self._exec
        ).pack(side="left")
        self.stop_button = ttk.Button(
            bar, text=get_string("🛑  Stop"), command=self._exec_stop, style="Danger.TButton"
        ).pack(side="right")

        self.exec_text = scrolledtext.ScrolledText(
            self.tab_exec,
            font=("Courier New", 10),
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="white",
            wrap="word",
            relief="flat",
        )
        self.exec_text.pack(fill="both", expand=True, padx=10, pady=8)

    def _exec(self):
        self.tcpclient = TCPClient(host="localhost", port=5000)
        try:
            saved_yaml_path = self._save_yaml()
            if not saved_yaml_path:
                self.status_var.set("Save and run aborted")
                self.exec_text.insert(tk.END, "Save and run aborted")
                return
            self.exec_text.delete("1.0", "end")

            def bg_task():
                env = {}
                uv_path = rf"""{Path("uv").with_suffix(".exe" if PLATFORM_SYSTEM == "Windows" else "")}"""
                lspicebatch_path = (
                    rf"""{Path(uv_path).parent.joinpath(LTSPICEBATCH)}"""
                )
                yaml_path = rf"""{Path(saved_yaml_path)}"""

                cmd_parts = [uv_path, "run", lspicebatch_path, "-c", yaml_path]
                if hasattr(self, "var_encode_only") and self.var_encode_only.get():
                    cmd_parts.append("--encode-only")
                if hasattr(self, "var_keep_images") and self.var_keep_images.get():
                    cmd_parts.append("--keep-images")
                if hasattr(self, "var_cleanup") and self.var_cleanup.get():
                    cmd_parts.append("--cleanup")
                if (
                    hasattr(self, "var_show_freq_domains")
                    and self.var_show_freq_domains.get()
                ):
                    cmd_parts.append("--show-freq-domains")
                if hasattr(self, "var_reset") and self.var_reset.get():
                    cmd_parts.append("--reset")

                if PLATFORM_SYSTEM == "Linux":
                    if self.var_wine_executable.get() != "":
                        env["WINEEXECUTABLE"] = self.var_wine_executable.get()
                    if self.var_wine_folder.get() != "":
                        env["WINEFOLDER"] = Path(self.var_wine_folder.get()).stem
                env = {**env, **os.environ.copy()}

                try:
                    process = subprocess.Popen(
                        cmd_parts,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        errors="replace",
                        env=env,
                        shell=False,
                    )

                    self.tcpclient.connect()

                    for line in process.stdout:
                        if re.search(r"^[\d\.%]+ ", line):
                            self.exec_text.replace("end-3l", "end-1l", line)
                        else:
                            self.exec_text.insert(tk.END, line)
                        self.exec_text.see(tk.END)

                    process.wait()

                except Exception as e:
                    self.exec_text.insert(
                        tk.END, f"""cmd: {" ".join(cmd_parts)}\nerror: {e}\n"""
                    )
                finally:
                    self.tcpclient.disconnect()
                    self.exec_text.see(tk.END)
                    self._save_config()

            thread = threading.Thread(target=bg_task, daemon=True)
            thread.start()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _exec_stop(self):
        try:
            self.tcpclient.send("stop")
        except OSError:
            print("could not send stop signal")
    
    # status bar
    def _build_status_bar(self):
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Button(
            bar,
            text=get_string("💾  Save YAML"),
            style="Success.TButton",
            command=self._save_yaml,
        ).pack(side="right", padx=5)
        ttk.Button(bar, text=get_string("📂  Load YAML"), command=self._load_yaml).pack(
            side="right", padx=5
        )
        ttk.Button(
            bar,
            text=get_string("🔄  Reset"),
            style="Danger.TButton",
            command=self._reset,
        ).pack(side="right", padx=5)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(bar, textvariable=self.status_var, style="TLabel").pack(
            side="left", padx=5
        )

    
    # collect data
    def _collect_data(self):
        data = {}

        # ffmpeg
        ffmpeg_bin = self.var_ffmpeg_bin.get().strip()
        if ffmpeg_bin:
            data["ffmpeg_bin"] = ffmpeg_bin

        framerate = self.var_ffmpeg_framerate.get().strip()
        if framerate:
            data["ffmpeg_framerate"] = self._cast(framerate)

        # image
        data["image"] = {
            "dpi": self._cast(self.var_image_dpi.get()),
            "width": self._cast(self.var_image_width.get()),
            "heigth": self._cast(self.var_image_heigth.get()),
        }

        # input_file
        inp = self.var_input_file.get().strip()
        if inp:
            data["input_file"] = inp

        # parallel / timeout
        data["parallel_sim"] = self._cast(self.var_parallel_sim.get())
        if data["parallel_sim"] == "":
            del data["parallel_sim"]
        data["parallel_plot"] = self._cast(self.var_parallel_plot.get())
        if data["parallel_plot"] == "":
            del data["parallel_plot"]
        data["runner_timeout"] = self._cast(self.var_runner_timeout.get())
        if data["runner_timeout"] == "":
            del data["runner_timeout"]

        # steps
        data["steps"] = [sf.collect() for sf in self.steps]

        # temp_folder
        tf = self.var_temp_folder.get().strip()
        if tf:
            data["temp_folder"] = tf

        return data

    @staticmethod
    def _cast(value):
        """try to cast as int/float, or keep string."""
        try:
            return int(value)
        except (ValueError, TypeError):
            pass
        try:
            return float(value)
        except (ValueError, TypeError):
            pass
        return value

    
    # generate yaml    
    def _to_yaml_string(self, data):
        header = "---\n# Generated by LTSpiceBatch - Config and Run\n"
        footer = "\n# end\n"

        temp = data.pop("temp_folder", None)

        yaml_body = yaml.dump(
            data,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            indent=2,
        )

        result = header + yaml_body
        if temp:
            result += f'\ntemp_folder: "{temp}"\n'
        result += footer

        if temp:
            data["temp_folder"] = temp

        return result

    
    # save yaml    
    def _save_yaml(self):
        save_file = False
        try:
            data = self._collect_data()
        except Exception as e:
            messagebox.showerror("Error collecting data", str(e))
            return
        if hasattr(self, "loaded_data"):
            if self.loaded_data == data:
                self.status_var.set("data didn't change, not saving")
                path = self.loaded_data_path
            else:
                save_file = True
        else:
            save_file = True
        if save_file:
            path = filedialog.asksaveasfilename(
                defaultextension=".yaml",
                filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
                title="Save YAML file",
            )
            if not path:
                return

            try:
                yaml_str = self._to_yaml_string(data)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(yaml_str)
                self.status_var.set(get_string(f"✅ Saved: {os.path.basename(path)}"))
            except Exception as e:
                messagebox.showerror("Save Error", str(e))
        return path

    
    # lod yaml    
    def _load_yaml(self):
        path = filedialog.askopenfilename(
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
            title="Load YAML file",
            initialdir=Path.home(),
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self.loaded_data_path = path
            self.loaded_data = data
            self._populate_from_data(data=data, reload_input=True)
            self.status_var.set(get_string(f"📂 Loaded: {os.path.basename(path)}"))
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _populate_from_data(self, data, reload_input=False):
        if not data:
            return
        # global vars
        self.var_ffmpeg_bin.set(data.get("ffmpeg_bin", ""))
        self.var_ffmpeg_framerate.set(str(data.get("ffmpeg_framerate", 6)))
        img = data.get("image", {})
        self.var_image_dpi.set(str(img.get("dpi", 100)))
        self.var_image_width.set(str(img.get("width", 1920)))
        self.var_image_heigth.set(str(img.get("heigth", 1080)))
        if reload_input:
            self.var_input_file.set(data.get("input_file", ""))
        self.var_parallel_sim.set(str(data.get("parallel_sim", 4)))
        self.var_parallel_plot.set(str(data.get("parallel_plot", 4)))
        self.var_runner_timeout.set(str(data.get("runner_timeout", 30)))
        self.var_temp_folder.set(data.get("temp_folder", ""))

        # steps
        for sf in self.steps[:]:
            sf.destroy()
        self.steps.clear()

        for step_data in data.get("steps", []):
            self._add_step(data=step_data)

    def _get_param_names(self):
        """list available params names in asc file"""
        avail_params = []
        input_file = getattr(self, "var_input_file", "").get().strip()
        if input_file == "":
            return []
        if not Path(input_file).exists():
            self.status_var.set(f"failed to load {input_file}")
            return []
        with open(input_file, "r", encoding=self._detect_encoding(input_file)) as f:
            lines = f.readlines()
        for line in lines:
            if re.search(r"\.param", line):
                for p in line.split("!")[1].split("\\n"):
                    m = re.match(
                        r"^\s*[^;]?\s*\.param\s+(?P<param_name>\w+)\s*=.+", p, re.I
                    )
                    if m:
                        avail_params.append(m.group("param_name").lower())
        return avail_params

    
    # reset    
    def _reset(self):
        if messagebox.askyesno("Reset", "Reset all fields to default values?"):
            self._populate_from_data(deepcopy(DEFAULT_CONFIG))
            self.status_var.set(get_string("🔄 Reset to defaults."))

    def _detect_encoding(self, file):
        detector = UniversalDetector()
        with open(file=file, mode="rb") as f:
            for line in f:
                detector.feed(line)
                if detector.done:
                    break
        result = detector.close()
        return result["encoding"]


# step widget
class StepFrame(ttk.Frame):
    def __init__(self, parent, app, index=0, data=None):
        super().__init__(parent, style="Card.TFrame", relief="groove", borderwidth=2)
        self.app = app
        self.index = index
        self.params = []  # ParamRow list
        self.traces = []  # TraceRow list

        self._build(data or deepcopy(DEFAULT_STEP))

    def set_index(self, i):
        self.index = i
        self.title_label.config(text=f"  Step #{i + 1}")

    def _build(self, data):
        # ---- header ----
        header = ttk.Frame(self, style="Card.TFrame")
        header.pack(fill="x", padx=5, pady=(5, 0))

        self.title_label = ttk.Label(
            header, text=f"  Step #{self.index + 1}", style="CardTitle.TLabel"
        )
        self.title_label.pack(side="left")

        # control buttons
        ttk.Button(
            header,
            text=get_string("🗑 Remove"),
            style="Danger.TButton",
            width=3,
            command=lambda: self.app._remove_step(self),
        ).pack(side="right", padx=2)
        ttk.Button(
            header,
            text=get_string("⬇ Down"),
            width=3,
            command=lambda: self.app._move_step_down(self),
        ).pack(side="right", padx=2)
        ttk.Button(
            header,
            text=get_string("⬆ Up"),
            width=3,
            command=lambda: self.app._move_step_up(self),
        ).pack(side="right", padx=2)

        # ---- body ----
        body = ttk.Frame(self, style="Card.TFrame")
        body.pack(fill="x", padx=10, pady=5)

        # line 1 : name
        row1 = ttk.Frame(body, style="Card.TFrame")
        row1.pack(fill="x", pady=3)
        ttk.Label(row1, text="name :", width=18, style="Card.TLabel", anchor="w").pack(
            side="left"
        )
        self.var_name = tk.StringVar(value=data.get("name", ""))
        ttk.Entry(row1, textvariable=self.var_name, width=40).pack(side="left")

        # line 2 : sim_command
        row2 = ttk.Frame(body, style="Card.TFrame")
        row2.pack(fill="x", pady=3)
        ttk.Label(
            row2, text="sim_command :", width=18, style="Card.TLabel", anchor="w"
        ).pack(side="left")
        self.var_sim = tk.StringVar(value=data.get("sim_command", ""))
        ttk.Entry(row2, textvariable=self.var_sim, width=40).pack(side="left")

        # line 3 : ffmpeg_framerate (optionnal by step)
        row3 = ttk.Frame(body, style="Card.TFrame")
        row3.pack(fill="x", pady=3)
        ttk.Label(
            row3,
            text="ffmpeg_framerate\n(optional) :",
            width=18,
            style="Card.TLabel",
            anchor="w",
        ).pack(side="left")
        self.var_framerate = tk.StringVar(
            value=str(data["ffmpeg_framerate"]) if "ffmpeg_framerate" in data else ""
        )
        ttk.Entry(row3, textvariable=self.var_framerate, width=10).pack(side="left")
        ttk.Label(
            row3, text="(leave empty to use global value)", style="Card.TLabel"
        ).pack(side="left", padx=5)

        # line 4 : fft
        row4 = ttk.Frame(body, style="Card.TFrame")
        row4.pack(fill="x", pady=3)
        ttk.Label(row4, text="fft :", width=18, style="Card.TLabel", anchor="w").pack(
            side="left"
        )
        self.var_fft = tk.BooleanVar(value=data.get("fft", False))
        ttk.Checkbutton(row4, variable=self.var_fft, onvalue=True, offvalue=False).pack(
            side="left"
        )
        ttk.Label(
            row4,
            text=f"""{" " * 15}fft_x_max :""",
            width=18,
            style="Card.TLabel",
            anchor="w",
        ).pack(side="left")
        self.var_fft_x_max = tk.StringVar(value=data.get("fft_x_max", 20000))
        ttk.Entry(row4, textvariable=self.var_fft_x_max, width=10).pack(side="left")

        # ---- traces ----
        lf_traces = ttk.LabelFrame(
            body, text=get_string(" 📈 tracestoplot "), padding=5
        )
        lf_traces.pack(fill="x", pady=5)

        self.traces_frame = ttk.Frame(lf_traces, style="Card.TFrame")
        self.traces_frame.pack(fill="x")

        btn_row = ttk.Frame(lf_traces, style="Card.TFrame")
        btn_row.pack(fill="x", pady=(3, 0))
        ttk.Button(
            btn_row, text=get_string("➕ Add Trace"), command=self._add_trace
        ).pack(side="left")

        for t in data.get("tracestoplot", []):
            self._add_trace(t)

        # ---- params ----
        lf_params = ttk.LabelFrame(body, text=get_string(" 🔧 parameters "), padding=5)
        lf_params.pack(fill="x", pady=5)

        ttk.Label(
            lf_params,
            text="Values: single (1e-9), list (10,73,259) or range {start, stop, step}",
            style="Card.TLabel",
        ).pack(anchor="w")

        self.params_frame = ttk.Frame(lf_params, style="Card.TFrame")
        self.params_frame.pack(fill="x")

        btn_row2 = ttk.Frame(lf_params, style="Card.TFrame")
        btn_row2.pack(fill="x", pady=(3, 0))
        ttk.Button(
            btn_row2, text=get_string("➕ Add Parameter"), command=self._add_param
        ).pack(side="left")

        for pname, pval in (data.get("parameters") or {}).items():
            self._add_param(pname, pval)

    # ---- traces ----
    def _add_trace(self, value=""):
        tr = TraceRow(self.traces_frame, self, value)
        tr.pack(fill="x", pady=1)
        self.traces.append(tr)

    def _remove_trace(self, tr):
        if tr in self.traces:
            self.traces.remove(tr)
        tr.destroy()

    # ---- params ----
    def _add_param(self, name="", value=None):
        pr = ParamRow(self.params_frame, self, name, value)
        pr.pack(fill="x", pady=1)
        self.params.append(pr)

    def _remove_param(self, pr):
        if pr in self.params:
            self.params.remove(pr)
        pr.destroy()

    # ---- collect vars ----
    def collect(self):
        step = {
            "name": self.var_name.get().strip(),
            "sim_command": self.var_sim.get().strip(),
            "fft": self.var_fft.get(),
        }
        fr = self.var_framerate.get().strip()
        if fr:
            step["ffmpeg_framerate"] = YamlGeneratorApp._cast(fr)
        if step["fft"]:
            step["fft_x_max"] = YamlGeneratorApp._cast(self.var_fft_x_max.get())

        step["tracestoplot"] = [
            t.get_value() for t in self.traces if t.get_value().strip()
        ]

        params = {}
        for p in self.params:
            k, v = p.collect()
            if k.strip():
                params[k.strip()] = v
        step["parameters"] = params if params else None

        return step


# traces widget
class TraceRow(ttk.Frame):
    def __init__(self, parent, step_frame, value=""):
        super().__init__(parent, style="Card.TFrame")
        self.step_frame = step_frame
        self.var = tk.StringVar(value=value)
        ttk.Label(self, text="trace :", width=10, style="Card.TLabel").pack(side="left")
        ttk.Entry(self, textvariable=self.var, width=30).pack(side="left", padx=3)
        ttk.Button(
            self,
            text=get_string("✕ Remove"),
            width=3,
            style="Danger.TButton",
            command=lambda: step_frame._remove_trace(self),
        ).pack(side="left")

    def get_value(self):
        return self.var.get()


# params widget
class ParamRow(ttk.Frame):
    """
    param can be one of:
      - single value : 6.8e-9
      - list         : 10, 73, 259, 657
      - range        : start / stop / step
    """

    MODES = ["Single value", "List", "Range (start/stop/step)"]

    def __init__(self, parent, step_frame, name="", value=None):
        super().__init__(parent, style="Card.TFrame")
        self.step_frame = step_frame
        self._build(name, value)

    def _build(self, name, value):
        # line 1 : name+ mode + rm button
        row1 = ttk.Frame(self, style="Card.TFrame")
        row1.pack(fill="x", pady=1)

        ttk.Label(row1, text="param name :", width=14, style="Card.TLabel").pack(
            side="left"
        )

        self.var_name = tk.StringVar()
        global param_names
        self.cb_name = ttk.Combobox(
            row1,
            textvariable=self.var_name,
            values=param_names,
            state="readonly",
            width=22,
        )
        self.cb_name.pack(side="left", padx=3)

        ttk.Label(row1, text="type :", style="Card.TLabel").pack(side="left")
        self.var_mode = tk.StringVar()
        self.cb_mode = ttk.Combobox(
            row1,
            textvariable=self.var_mode,
            values=self.MODES,
            state="readonly",
            width=22,
        )
        self.cb_mode.pack(side="left", padx=3)
        self.cb_mode.bind("<<ComboboxSelected>>", self._on_mode_change)

        ttk.Button(
            row1,
            text=get_string("✕ Remove"),
            width=3,
            style="Danger.TButton",
            command=lambda: self.step_frame._remove_param(self),
        ).pack(side="left", padx=5)

        # line 2 : values (dynamic)
        self.row2 = ttk.Frame(self, style="Card.TFrame")
        self.row2.pack(fill="x", pady=1, padx=20)

        # init depending on value type
        if isinstance(value, dict):
            self.var_mode.set("Range (start/stop/step)")
            self._build_range_widgets(value)
        elif isinstance(value, list):
            self.var_mode.set("List")
            self._build_list_widgets(value)
        else:
            self.var_mode.set("Single value")
            self._build_single_widgets(str(value) if value is not None else "")

        if name.lower() in param_names:
            self.var_name.set(name)

    def _clear_row2(self):
        for w in self.row2.winfo_children():
            w.destroy()

    def _on_mode_change(self, event=None):
        mode = self.var_mode.get()
        self._clear_row2()
        if mode == "Single value":
            self._build_single_widgets()
        elif mode == "List":
            self._build_list_widgets()
        else:
            self._build_range_widgets()

    def _build_single_widgets(self, val=""):
        ttk.Label(self.row2, text="value :", width=8, style="Card.TLabel").pack(
            side="left"
        )
        self.var_single = tk.StringVar(value=val)
        ttk.Entry(self.row2, textvariable=self.var_single, width=20).pack(side="left")

    def _build_list_widgets(self, lst=None):
        ttk.Label(
            self.row2, text="values (comma separated) :", width=26, style="Card.TLabel"
        ).pack(side="left")
        default = ", ".join(str(v) for v in lst) if lst else ""
        self.var_list = tk.StringVar(value=default)
        ttk.Entry(self.row2, textvariable=self.var_list, width=40).pack(side="left")

    def _build_range_widgets(self, d=None):
        d = d or {}
        for lbl, attr, default in [
            ("start :", "var_start", str(d.get("start", ""))),
            ("stop :", "var_stop", str(d.get("stop", ""))),
            ("step :", "var_step", str(d.get("step", ""))),
        ]:
            ttk.Label(self.row2, text=lbl, width=6, style="Card.TLabel").pack(
                side="left"
            )
            var = tk.StringVar(value=default)
            setattr(self, attr, var)
            ttk.Entry(self.row2, textvariable=var, width=12).pack(
                side="left", padx=(0, 8)
            )

    def collect(self):
        name = self.var_name.get()
        mode = self.var_mode.get()

        if mode == "Single value":
            raw = self.var_single.get().strip()
            value = YamlGeneratorApp._cast(raw)

        elif mode == "List":
            raw = self.var_list.get()
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            value = [YamlGeneratorApp._cast(p) for p in parts]

        else:  # range
            value = {
                "start": YamlGeneratorApp._cast(self.var_start.get().strip()),
                "stop": YamlGeneratorApp._cast(self.var_stop.get().strip()),
                "step": YamlGeneratorApp._cast(self.var_step.get().strip()),
            }

        return name, value


if __name__ == "__main__":
    app = YamlGeneratorApp()
    app.mainloop()
