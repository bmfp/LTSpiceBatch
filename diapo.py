#!/usr/bin/env python3
"""Images slideshow"""

import json
import os
import glob
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk


class SlideshowApp(tk.Tk):
    """Slideshow app with preloading"""

    def __init__(self):
        super().__init__()
        self.geometry("1024x768")
        self.title("LTSpiceBatch Slideshow")
        

        # --- Slideshow status ---
        self.current_step = ""
        self.images_files = []
        self.images_files_props = {}
        self.images_files_filtered = []
        self.steps = []
        self.step_params = {}
        self.current_index = 0
        self.playing = False
        self.interval_ms = 1000
        self.interval_ms_min = int(1000 / 24)
        self.interval_ms_max = 2000
        self.timer_id = None

        # --- Preload: images pool ---
        self.preload_count = 2
        self._pool = {}
        self.current_photo = None
        self.canvas_img_id = None

        self.interval_var = tk.IntVar(value=self.interval_ms)

        # --- Interface ---
        self._build_styles()
        self._build_ui()
        self.bind("<Escape>", lambda e: self._toggle_fullscreen(False))
        self.is_fullscreen = False

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

    # ==================================================================
    # Interface build
    # ==================================================================
    def _build_ui(self):
        """Creates widgets and layout"""
        main = ttk.Frame(self, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # Image canvas
        self.canvas = tk.Canvas(main, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Resize canvas on window resize
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Image Filters
        self.filters = ttk.Frame(main)
        self.filters.pack(fill=tk.X, pady=(10, 0))

        # Controls bar
        ctrl = ttk.Frame(main)
        ctrl.pack(fill=tk.X, pady=(10, 0))

        def make_btn(text, command):
            return ttk.Button(ctrl, text=text, command=command)

        self.btn_prev   = make_btn("◀ Previous", self._prev)
        self.btn_play   = make_btn("▶ Play",     self._toggle_play)
        self.btn_next   = make_btn("Next ▶",     self._next)
        self.btn_fs     = make_btn("Fullscreen",  self._toggle_fullscreen)
        self.btn_folder = make_btn("Directory…",  self._browse_folder)

        self.btn_prev.pack(side=tk.LEFT, padx=4)
        self.btn_play.pack(side=tk.LEFT, padx=4)
        self.btn_next.pack(side=tk.LEFT, padx=4)
        self.btn_fs.pack(side=tk.RIGHT, padx=4)
        self.btn_folder.pack(side=tk.RIGHT, padx=4)

        ttk.Label(ctrl, text="Interval (ms)").pack(side=tk.LEFT, padx=(30, 4))
        ttk.Spinbox(
            ctrl, from_=self.interval_ms_min, to=self.interval_ms_max, increment=250,
            width=8, textvariable=self.interval_var,
            command=self._on_interval_change,
            name="spinbox_interval"
        ).pack(side=tk.LEFT)

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")

    # ==================================================================
    # Directory load
    # ==================================================================
    def _browse_folder(self):
        """Open directory and load png images."""
        try:
            folder = Path(filedialog.askdirectory(title="Choose images directory", mustexist=True))
        except TypeError:
            return

        try:
            for imglist in list(folder.glob("*_imglist.txt")):
                with open(imglist, "r") as f:
                    self.images_files += [str(Path.joinpath(folder, Path(line.strip().split(" ")[1]))) for line in f.readlines() if line.startswith("file")]
        except IndexError:
            pattern = os.path.join(folder, "*.png")
            self.images_files = sorted(glob.glob(pattern))

        if not self.images_files:
            messagebox.showwarning("No image",
                "No png image in directory")
            return

        _fft_images = []
        for img in self.images_files:
            if "fft_" in img:
                _fft_images.append(img.replace("fft_", "", 1))

        self.images_files = _fft_images + self.images_files
        self.images_files_filtered = self.images_files
        self.images_files_props = {img: {} for img in self.images_files}

        # Load images properties
        for f in self.images_files:
            with Image.open(f) as img:
                try:
                    self.images_files_props[f] = json.loads(img.info["Description"])
                except (json.JSONDecodeError, KeyError):
                    del self.images_files_props[f]

        if len(self.images_files_props) == 0:
            messagebox.showwarning("No valid image",
                "No image with valid metadata")
            return

        self.state('normal')
        self.is_fullscreen = False
        self.w_h = (self.canvas.winfo_width(), self.canvas.winfo_height())

        self.current_index = 0
        self._aggr_props()
        self._clear_pool()
        self._show_current()

    # ==================================================================
    # Navigation
    # ==================================================================
    def _prev(self):
        if not self.images_files_filtered:
            return
        self.current_index = (self.current_index - 1) % len(self.images_files_filtered)
        self._clear_pool()
        self._show_current()

    def _next(self):
        if not self.images_files_filtered:
            return
        self.current_index = (self.current_index + 1) % len(self.images_files_filtered)
        self._clear_pool()
        self._show_current()

    # ==================================================================
    # Play / Pause
    # ==================================================================
    def _toggle_play(self):
        if not self.images_files_filtered:
            return
        self.playing = not self.playing
        if self.playing:
            self.btn_play.config(text="⏸ Pause")
            self._schedule_next()
        else:
            self.btn_play.config(text="▶ Lecture")
            if self.timer_id:
                self.after_cancel(self.timer_id)
                self.timer_id = None

    def _schedule_next(self):
        """Planifie l'image suivante après interval_ms."""
        if not self.playing or not self.images_files_filtered:
            return
        n = len(self.images_files_filtered)
        self.current_index = (self.current_index + 1) % n
        self._clear_pool()
        self._show_current()
        self.timer_id = self.after(
            self.interval_ms, self._schedule_next
        )

    # ==================================================================
    # Display + preload
    # ==================================================================
    def _show_current(self):
        """Display current image and preload next"""
        if not self.images_files_filtered:
            return

        n = len(self.images_files_filtered)
        idx = self.current_index

        photo, _ = self._get_or_load(idx)

        if self.canvas_img_id is None:
            self.canvas_img_id = self.canvas.create_image(
                0, 0, anchor=tk.NW, image=photo
            )
        else:
            self.canvas.itemconfig(self.canvas_img_id, image=photo)

        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        self.current_photo = photo

        # Preload next images in pool
        for offset in range(1, self.preload_count + 1):
            pre_idx = (idx + offset) % n
            if pre_idx not in self._pool:
                _, orig = self._get_or_load(pre_idx)
                self._pool[pre_idx] = {"original": orig}

    def _get_or_load(self, idx):
        """Returns (PhotoImage, original_pil) for index"""
        n = len(self.images_files_filtered)
        path = self.images_files_filtered[idx % n]
        pil_img = Image.open(path)

        pil_img_ratio = pil_img.width / pil_img.height
        w_h_ratio = self.w_h[0] / self.w_h[1]
        if pil_img_ratio >= w_h_ratio:
            self.w_h = (self.w_h[0], int(self.w_h[0]/pil_img_ratio))
        else:
            self.w_h = (int(self.w_h[1]*pil_img_ratio), self.w_h[1])

        pil_resized = pil_img.resize(self.w_h, Image.LANCZOS)
        
        new_photo = ImageTk.PhotoImage(pil_resized)

        return new_photo, pil_img

    def _clear_pool(self):
        """Clear all images from pool and canvas (all but current)"""
        if self.canvas_img_id is not None:
            try:
                self.canvas.delete(self.canvas_img_id)
            except tk.TclError:
                pass
            self.canvas_img_id = None

        self._pool.clear()

    def _var_change_callback(self, *args):
        """React to interface changes"""
        step = self.vars["spinbox_steps_value"].get()
        setattr(self, "current_step", step)
        if args[0] == "spinbox_steps":
            self._aggr_props(step=step)
        if "from" in args[0] or "to" in args[0]:
            values = self.vars[args[0].replace("_from_value", "_values").replace("_to_value", "_values")]
            selected_value = self.vars[args[0]].get()
        if "from" in args[0]:
            combobox_to: ttk.Combobox = self.vars[args[0].replace("_from_value", "_to_combobox")]
            combobox_to.configure(values=values[values.index(selected_value):])
        elif "to" in args[0]:
            combobox_from: ttk.Combobox = self.vars[args[0].replace("_to_value", "_from_combobox")]
            combobox_from.configure(values=values[:values.index(selected_value)+1])

        self.images_files_filtered = [img for img in self.images_files]
        for img in self.images_files:
            if self.images_files_props[img]["step"] != step:
                try:
                    self.images_files_filtered.remove(img)
                except ValueError:
                    pass
                continue
            for params in self.images_files_props[img]["params"]:
                for param in params:
                    try:
                        limit_from = self.vars[f"spinbox_{param}_from_combobox"].cget("values")
                        limit_to = self.vars[f"spinbox_{param}_to_combobox"].cget("values")
                    except KeyError:
                        continue
                    if len(limit_from) == len(limit_to):
                        continue
                    limits = list(set(limit_from) & set(limit_to))
                    val = params[param]
                    if not val in limits:
                        try:
                            self.images_files_filtered.remove(img)
                        except ValueError:
                            pass
        self.current_index = -1
        self._next()

    def _aggr_props(self, step=None):
        """Aggregate steps and parameters"""
        if step is not None:
            self.step_params = {}
        for _, props in self.images_files_props.items():
            if step is not None and props["step"] != step:
                continue

            if props["step"] not in self.steps:
                self.steps.append(props["step"])
            for params in props["params"]:
                for param in params:
                    if param not in self.step_params:
                        self.step_params[param] = []
                    if isinstance(params[param], (int, float)) and params[param] not in self.step_params[param]:
                        self.step_params[param].append(params[param])

        # clean filters frame
        for widget in self.filters.winfo_children():
            widget.destroy()
        self.vars = {}
        self.vars["spinbox_steps_value"] = tk.StringVar(self.filters, getattr(self, "current_step", self.steps[0]), "spinbox_steps")
        self.vars["spinbox_steps_value"].trace_add("write", self._var_change_callback)
        self.vars["spinbox_steps_combobox"] = ttk.Combobox(self.filters, values=self.steps, state="readonly", textvariable=self.vars["spinbox_steps_value"])
        self.vars["spinbox_steps_combobox"].pack(side=tk.LEFT, padx=(0, 4))
        for param in self.step_params:
            if self.step_params[param] == []:
                continue
            sorted_values = self.step_params[param]
            self.vars[f"spinbox_{param}_from_value"] = tk.DoubleVar(self.filters, sorted_values[0],  f"spinbox_{param}_from_value")
            self.vars[f"spinbox_{param}_from_value"].trace_add("write", self._var_change_callback)
            self.vars[f"spinbox_{param}_values"] = sorted_values
            self.vars[f"spinbox_{param}_to_value"] = tk.DoubleVar(self.filters, sorted_values[-1],  f"spinbox_{param}_to_value")
            self.vars[f"spinbox_{param}_to_value"].trace_add("write", self._var_change_callback)
            ttk.Label(self.filters, text=param).pack(side=tk.LEFT, padx=(0, 4))
            self.vars[f"spinbox_{param}_from_combobox"] = ttk.Combobox(self.filters, values=sorted_values, state="readonly", textvariable=self.vars[f"spinbox_{param}_from_value"])
            self.vars[f"spinbox_{param}_from_combobox"].pack(side=tk.LEFT, padx=(0, 4))
            ttk.Label(self.filters, text="->").pack(side=tk.LEFT, padx=(0, 4))
            self.vars[f"spinbox_{param}_to_combobox"] = ttk.Combobox(self.filters, values=sorted_values, state="readonly", textvariable=self.vars[f"spinbox_{param}_to_value"])
            self.vars[f"spinbox_{param}_to_combobox"].pack(side=tk.LEFT, padx=(0, 4))

    # ==================================================================
    # Fullscreen
    # ==================================================================
    def _toggle_fullscreen(self, _=None):
        """Toggle between fullscreen and windowed"""
        if not self.is_fullscreen:
            self.attributes('-fullscreen', True)
            self.canvas.config(cursor="none")
            self.is_fullscreen = True
        else:
            self.attributes('-fullscreen', False)
            self.state('normal')
            self.canvas.config(cursor="")
            self.is_fullscreen = False

    # ==================================================================
    # Tempo
    # ==================================================================
    def _on_interval_change(self, _=None):
        """Update interval from spinbox"""
        try:
            new_val = int(float(self.interval_var.get()))
            if self.interval_ms_min <= new_val <= self.interval_ms_max:
                self.interval_ms = new_val
        except (ValueError, tk.TclError):
            pass

    # ==================================================================
    # Keyboard shortcuts
    # ==================================================================
    def _bind_keys(self):
        """Global keyboard shortcuts"""
        self.bind("<Left>",  lambda e: self._prev())
        self.bind("<Right>", lambda e: self._next())
        self.bind("<space>", lambda e: self._toggle_play())
        self.bind("f",       lambda e: self._toggle_fullscreen())

    def _on_canvas_resize(self, event):
        """Resize current image to fit canvas"""
        if not self.images_files or self.current_photo is None:
            return
        
        # Avail canvas dimensions
        canvas_w = event.width
        canvas_h = event.height
        
        if canvas_w <= 0 or canvas_h <= 0:
            return
        
        # Original image dimensions
        img_w = self.current_photo.width()
        img_h = self.current_photo.height()
        
        # Get proprtions ratio
        ratio = min(canvas_w / img_w, canvas_h / img_h)
        self.w_h = (int(img_w * ratio), int(img_h * ratio))

        if self.w_h[0] <= 0 or self.w_h[1] <= 0:
            return
        
        # Resize image
        original_path = self.images_files[self.current_index]
        pil_img = Image.open(original_path)
        pil_resized = pil_img.resize(self.w_h, Image.LANCZOS)
        
        new_photo = ImageTk.PhotoImage(pil_resized)
        
        # Update image in canvas
        if self.canvas_img_id is None:
            self.canvas_img_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=new_photo)
        else:
            self.canvas.itemconfig(self.canvas_img_id, image=new_photo)

        self.current_photo = new_photo
        
        # Adjust canvas scroll zone
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def _on_mousewheel(self, event: tk.Event):
        try:
            if event.widget.winfo_name() == "spinbox_interval":
                return
        except AttributeError:
            return
        if event.num == 4:   # Linux up
            self._next()
        elif event.num == 5: # Linux down
            self._prev()
        else:
            if event.delta > 0:
                self._next()
            else:
                self._prev()

def main():
    app = SlideshowApp()
    app._bind_keys()
    app.mainloop()


if __name__ == "__main__":
    main()
