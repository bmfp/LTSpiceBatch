import argparse
import concurrent.futures
import math
import os
import signal
import string
import subprocess
import sys
import tempfile
from datetime import datetime
from glob import glob
from pathlib import Path
from subprocess import CalledProcessError, check_output

import matplotlib.pyplot as plt
import numpy
import yaml
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from PyLTSpice import (LTspice, RawRead, SimRunner, SpiceEditor,
                       SpiceReadException)
from unidecode import unidecode


import socket
import threading


class TCPServer:
    def __init__(self, host="localhost", port=5000, mainpid=-1):
        print(f"TCPServer:__init__ host={host}, port={port}, mainpid={mainpid}")
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        self.mainpid = mainpid

    def start(self):
        """Démarre le serveur et écoute les connexions."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Réutiliser l'adresse si le serveur redémarre
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.running = True
        # print(f"[SERVEUR] En écoute sur {self.host}:{self.port}")

        while self.running:
            try:
                client_socket, address = self.server_socket.accept()
                # print(f"[SERVEUR] Connexion acceptée de {address}")
                # Gérer chaque client dans un thread séparé
                thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, address)
                )
                thread.daemon = True
                thread.start()
            except OSError:
                break  # Le serveur a été arrêté

    def handle_client(self, client_socket, address):
        """Traite les messages d'un client connecté."""
        with client_socket:
            while True:
                try:
                    # Recevoir les données (buffer de 1024 octets)
                    data = client_socket.recv(1024)
                    if not data:
                        # print(f"[SERVEUR] Client {address} déconnecté.")
                        break

                    message = data.decode("utf-8")
                    # print(f"[SERVEUR] Reçu de {address} : {message}")

                    # Envoyer une réponse
                    response = f"Message bien reçu : '{message}'"
                    client_socket.sendall(response.encode("utf-8"))

                    if message == "stop":
                        # print("received stop")
                        os.kill(self.mainpid, signal.SIGINT)

                except ConnectionResetError:
                    print(f"[SERVEUR] Connexion perdue avec {address}.")
                    break

    def stop(self):
        """Arrête le serveur."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        # print("[SERVEUR] Arrêté.")

class LTSpiceBatch(object):
    def __init__(self, args: argparse.Namespace) -> None:
        cpu_count = os.cpu_count()
        self.config = self.load_config(config_file=args.config)
        self.step = {}
        self.debug = self.config.get("debug", False)
        self.parallel_sim = self.config.get("parallel_sim", int(cpu_count / 2))
        self.parallel_plot = self.config.get(
            "parallel_plot", cpu_count - 2 if cpu_count > 2 else 1
        )
        self.temp_folder = Path(
            self.config.get("temp_folder", f"{tempfile.gettempdir()}/LTSpiceBatch")
        )
        self.imglist_file = (
            f"{self.temp_folder}/{datetime.timestamp(datetime.now())}_imglist.txt"
        )
        self.input_file = self.config["input_file"]
        self.output_file = ""
        self.ffmpeg_bin = Path(
            self.config.get(
                "ffmpeg_bin",
                Path(__file__).parent.joinpath(
                    f"""ffmpeg{".exe" if os.name == "nt" else ""}"""
                ),
            )
        )
        self.ffmpeg_framerate = self.config.get("ffmpeg_framerate", 1)
        self.ffmpeg_hw_accel = self.config.get("ffmpeg_hw_accel", False)
        self.runner_timeout = self.config.get("runner_timeout", 60)
        self.colors = [
            "#e60049",
            "#0bb4ff",
            "#50e991",
            "#e6d800",
            "#9b19f5",
            "#ffa300",
            "#dc0ab4",
            "#b3d4ff",
            "#00bfa0",
        ]
        self.freq_domains = {
            "sub": [1, 20, "red"],
            "lo-lo": [20, 40, "orange"],
            "lo": [40, 160, "yellow"],
            "lo-mid": [160, 315, "green"],
            "mid": [315, 2500, "blue"],
            "hi-mid": [2500, 5000, "indigo"],
            "hi": [5000, 10000, "purple"],
            "hi-hi": [10000, 20000, "white"],
        }
        self.start = None
        self.reset = args.reset
        self.show_freq_domains = args.show_freq_domains
        self.imglist = []
        self.data_to_analyze = []
        self.data_analyzed = 0
        self.encode_only = args.encode_only

        try:
            self.term_width = os.get_terminal_size().columns
        except OSError:
            # not in a shell
            self.term_width = 80

        if not self.temp_folder.exists():
            os.mkdir(self.temp_folder)
        try:
            assert os.access(self.temp_folder, os.W_OK)
        except AssertionError:
            self.logprint(f"Make sure {self.temp_folder} is writable")
            sys.exit(1)

        self.logprint(f"""launching with:
debug: {self.debug}
parallel_sim: {self.parallel_sim}
parallel_plot: {self.parallel_plot}
temp_folder: {self.temp_folder}
input_file: {self.input_file}
ffmpeg_bin: {self.ffmpeg_bin}
ffmpeg_framerate: {self.ffmpeg_framerate}
ffmpeg_hw_accel: {self.ffmpeg_hw_accel}
runner_timeout: {self.runner_timeout}
reset: {self.reset}
show_freq_domains: {self.show_freq_domains}
""")
        self.server = TCPServer(host="localhost", port=5000, mainpid=os.getpid())
        server_thread = threading.Thread(target=self.server.start, daemon=True)
        server_thread.start()
        

    def load_config(self, config_file) -> dict:
        """
        load yml into config dict
        """
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

    def logprint(self, msg):
        print(f"[{datetime.now()}] - {msg}")

    def print_sameline(self, cur, total):
        """
        print progress using one term line
        """

        sys.stdout.write(self.term_width * " " + "\r")
        sys.stdout.write(f"{cur * 100 / total:.1f}% ({cur}/{total})\r")
        sys.stdout.flush()

    @staticmethod
    def opposite_sign(x, y) -> bool:
        return (y >= 0) if (x < 0) else (y < 0)

    def set_ffmpeg_args(
        self, format="concat", input=None, ffmpeg_hw_accel=False
    ) -> list:
        """
        test cuda availability
        """
        ffmpeg_hw_accel_args = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        verbosity_args = (
            ["-stats"]
            if self.debug
            else [
                "-v",
                "quiet",
            ]
        )
        args = [
            "-an",
            "-sn",
            "-y",
            "-f",
            format,
            "-i",
            self.imglist_file if input is None else input,
            "-c:v",
            "h264_nvenc" if ffmpeg_hw_accel else "libopenh264",
            "-r",
            str(self.step.get("ffmpeg_framerate", self.ffmpeg_framerate)),
            "-pix_fmt",
            "yuv420p",
        ]
        return (
            [*ffmpeg_hw_accel_args, *verbosity_args, *args]
            if ffmpeg_hw_accel
            else [*verbosity_args, *args]
        )

    def test_ffmpeg(self):
        try:
            assert os.access(self.ffmpeg_bin, os.X_OK)
        except AssertionError:
            self.logprint(f"Make sure {self.ffmpeg_bin} is executable")
            sys.exit(1)

        # test encoding with nvenc
        ffmpeg_args = self.set_ffmpeg_args(
            format="lavfi",
            input="testsrc=duration=3:size=320x240:rate=1",
            ffmpeg_hw_accel=True,
        )
        try:
            check_output(
                [self.ffmpeg_bin, *ffmpeg_args, f"{self.temp_folder}/test.mp4"]
            )
            self.ffmpeg_hw_accel = True
            self.logprint("nvenc enabled")
        except CalledProcessError:
            # test encoding without nvenc
            ffmpeg_args = self.set_ffmpeg_args(
                format="lavfi",
                input="testsrc=duration=3:size=320x240:rate=1",
                ffmpeg_hw_accel=False,
            )
            check_output(
                [self.ffmpeg_bin, *ffmpeg_args, f"{self.temp_folder}/test.mp4"]
            )
            self.ffmpeg_hw_accel = False
            self.logprint("nvenc disabled")

    def to_db(self, arr):
        """
        convert cartesian values array to dB
        """
        return [20 * math.log(numpy.abs(a), 10) for a in arr]

    def to_deg(self, arr):
        """
        convert cartesian values array to phase shift in degrees
        """
        out = []
        prev = math.atan2(numpy.imag(arr[0]), numpy.real(arr[0])) * 180 / math.pi
        cycle = 0
        for a in arr:
            degs = math.atan2(numpy.imag(a), numpy.real(a)) * 180 / math.pi
            if self.opposite_sign(degs, prev) and degs >= prev:
                cycle += 1
            elif self.opposite_sign(degs, prev) and degs < prev and prev - degs > 180:
                cycle -= 1
            out.append(degs - cycle * 360)
            prev = degs
        return out

    def product(self, ar_list):
        """
        flatten iterables
        """
        if not ar_list:
            yield ()
        else:
            for a in ar_list[0]:
                for prod in self.product(ar_list[1:]):
                    yield (a,) + prod

    def process_data_ac(self, raw_file, tracestoplot):
        """
        This is the function that will process the data from AC analysis simulations
        """
        raw_file_path = Path(raw_file)
        raw = RawRead(raw_filename=raw_file_path, traces_to_read="*")
        # raw.to_csv(filename="E:/CAD/splitter-mixer/output.csv", columns=["V(In)", "V(Out)"], separator=";")

        steps = raw.get_steps()
        mag: Axes
        fig, mag = plt.subplots(
            figsize=(
                self.config["image"]["width"] / self.config["image"]["dpi"],
                self.config["image"]["heigth"] / self.config["image"]["dpi"],
            ),
            dpi=self.config["image"]["dpi"],
            layout="tight",
        )
        mag.set_facecolor(color="black")
        plt.title("Step: " + self.step["name"])

        mag.set_xlabel("Frequency (Hz)")
        mag.set_ylabel("Amplitude (dB)")
        mag.set_xlim(
            left=numpy.real(raw.get_axis()[0]), right=numpy.real(raw.get_axis()[-1])
        )
        mag.set_ylim(
            bottom=self.step["mag_y_min"] if "mag_y_min" in self.step else -21,
            top=self.step["mag_y_max"] if "mag_y_max" in self.step else 12,
        )
        mag.yaxis.label.set_color("blue")
        mag.tick_params(axis="y", color="blue")
        mag.grid(which="both", color="blue")
        for i, step in enumerate(steps):
            for j, trace in enumerate(tracestoplot):
                mag.semilogx(
                    numpy.real(raw.get_axis()),
                    numpy.apply_along_axis(
                        self.to_db, 0, raw.get_trace(trace).get_wave(step=step)
                    ),
                    color=self.colors[i * j + j],
                    lw=2,
                    label=f"""{self.step["name"]} - {trace}""",
                )
        mag.legend(loc="upper left")

        phase: Axes = mag.twinx()
        phase.set_ylabel("Phase (deg)")
        phase.set_ylim(
            bottom=self.step["phase_y_min"] if "phase_y_min" in self.step else -360,
            top=self.step["phase_y_max"] if "phase_y_max" in self.step else 0,
        )
        phase.yaxis.label.set_color("green")
        phase.tick_params(axis="y", color="green")
        phase.grid(which="both", color="green")
        for i, step in enumerate(steps):
            for j, trace in enumerate(tracestoplot):
                phase.semilogx(
                    numpy.real(raw.get_axis()),
                    numpy.apply_along_axis(
                        self.to_deg, 0, raw.get_trace(trace).get_wave(step=step)
                    ),
                    color=self.colors[i * j + i],
                    ls="dotted",
                    lw=2,
                    label=f"""{self.step["name"]} - {trace}""",
                )
        phase.legend(loc="upper right")

        text_lines = []
        for param in raw_file_path.stem.split("-")[1:]:
            param_name = "_".join(param.split("_")[:-1])
            try:
                param_value = int(param.split("_")[-1])
            except ValueError:
                try:
                    param_value = float(param.split("_")[-1])
                except ValueError:
                    param_value = param.split("_")[-1]
            text_lines.append(f"""{param_name} = {param_value}""")
        text = "\n".join(text_lines)

        pt = phase.text(
            0, 0, text, fontsize=14, color="white", transform=phase.transAxes
        )
        r = fig.canvas.get_renderer()
        fig.canvas.draw()
        bbox = pt.get_window_extent(renderer=r).transformed(phase.transAxes.inverted())
        x0 = bbox.x0
        y0 = bbox.y0
        x1 = bbox.x1
        y1 = bbox.y1
        phase.add_patch(
            Rectangle(
                (x0, y0),
                width=x1 - x0,
                height=y1 - y0,
                transform=phase.transAxes,
                color="dimgray",
                alpha=0.5,
            )
        )
        prev = x0
        y1 = (
            phase.get_window_extent(renderer=r)
            .transformed(phase.transAxes.inverted())
            .y1
            - y1
        ) / 5
        if self.show_freq_domains:
            for dom in self.freq_domains:
                x0 = numpy.real(raw.get_axis()[0])
                x1 = numpy.real(raw.get_axis()[-1])
                f1 = (
                    math.log10(self.freq_domains[dom][0])
                    if self.freq_domains[dom][0] >= x0
                    else math.log10(x0)
                )
                width = (
                    (math.log10(self.freq_domains[dom][1]) - f1)
                    / (math.log10(x1) - math.log10(x0))
                    if self.freq_domains[dom][1] <= x1
                    else (math.log10(x1) - f1) / (math.log10(x1) - math.log10(x0))
                )
                # print(f"dom: {dom}, x0: {x0},  x1: {x1},  f1: {f1},  width: {width}")
                if width <= 0:
                    continue
                phase.add_patch(
                    Rectangle(
                        xy=(prev, y0),
                        width=width,
                        height=y1 - y0,
                        transform=phase.transAxes,
                        color=self.freq_domains[dom][2],
                        alpha=0.2,
                    )
                )
                phase.text(
                    prev + width / 2,
                    (y1 - y0) / 2,
                    dom,
                    fontsize=14,
                    color="white",
                    transform=phase.transAxes,
                )
                prev += width

        try:
            filename = Path.joinpath(
                raw_file_path.parent, raw_file_path.name
            ).with_suffix(".png")
        except Exception as e:
            print("filenameException", e)
        fig.savefig(filename)
        plt.close()
        return

    def process_data_tran_fft(self, raw_file, tracestoplot):
        """
        This is the function that will process the data from transient analysis simulations
        """
        raw_file_path = Path(raw_file)
        raw = RawRead(raw_filename=raw_file_path, traces_to_read="*")

        steps = raw.get_steps()
        ax: Axes
        fig, ax = plt.subplots(
            figsize=(
                self.config["image"]["width"] / self.config["image"]["dpi"],
                self.config["image"]["heigth"] / self.config["image"]["dpi"],
            ),
            dpi=self.config["image"]["dpi"],
            layout="tight",
        )
        ax.set_facecolor(color="black")
        plt.title("Step: " + self.step["name"])

        ax.set_xlabel("Freq (Hz)")
        ax.set_ylabel("Rel count (pct)")
        ax.tick_params(axis="y", which="both")
        ax.grid(which="both", color="white")

        fft_result_maxes = []
        fft_data = {
            s: {t: {"axis": None, "fft_result": None} for t in tracestoplot}
            for s in steps
        }
        # print(fft_data)
        for i, step in enumerate(steps):
            for j, trace in enumerate(tracestoplot):
                raw_get_axis_ori = raw.get_axis(step=step)
                wave_ori = raw.get_trace(trace).get_wave(step=step)

                raw_get_axis = numpy.linspace(
                    raw_get_axis_ori[0], raw_get_axis_ori[-1], len(raw_get_axis_ori)
                )
                wave = numpy.interp(raw_get_axis, raw_get_axis_ori, wave_ori)
                fft_result = numpy.absolute(numpy.fft.fft(wave))
                fft_result_max = numpy.max(fft_result)
                fft_result_maxes.append(fft_result_max)
                try:
                    fftfreq2 = numpy.fft.fftfreq(
                        len(fft_result), raw_get_axis[1] - raw_get_axis[0]
                    )
                except Exception as e:
                    print("fftfreqException\n", e)

                fft_data[step][trace]["fftfreq2"] = fftfreq2
        fft_result_max_overall = numpy.max(fft_result_maxes)

        for i, step in enumerate(steps):
            for j, trace in enumerate(tracestoplot):
                fft_result_as_pct = fft_result / fft_result_max_overall * 100
                ax.plot(
                    fft_data[step][trace]["fftfreq2"][
                        : len(fft_data[step][trace]["fftfreq2"]) // 2
                    ],
                    fft_result_as_pct[: len(fft_data[step][trace]["fftfreq2"]) // 2],
                    color=self.colors[i * j + j],
                    lw=2,
                    label=f"""{self.step["name"]} - {trace}""",
                )
        yabs_max = abs(max(ax.get_ylim(), key=abs))
        ax.set_xlim(
            xmin=0,
            xmax=self.step["fft_x_max"] if "fft_x_max" in self.step else 20e3,
        )
        ax.set_ylim(
            ymin=-1,
            ymax=self.step["y_max"] if "y_max" in self.step else yabs_max,
        )
        ax.legend(loc="upper left")

        text_lines = []
        for param in raw_file_path.stem.split("-")[1:]:
            param_name = "_".join(param.split("_")[:-1])
            try:
                param_value = int(param.split("_")[-1])
            except ValueError:
                try:
                    param_value = float(param.split("_")[-1])
                except ValueError:
                    param_value = param.split("_")[-1]
            text_lines.append(f"""{param_name} = {param_value}""")
        text = "\n".join(text_lines)

        pt = ax.text(0, 0, text, fontsize=14, color="white", transform=ax.transAxes)
        r = fig.canvas.get_renderer()
        fig.canvas.draw()
        bbox = pt.get_window_extent(renderer=r).transformed(ax.transAxes.inverted())
        x0 = bbox.x0
        y0 = bbox.y0
        x1 = bbox.x1
        y1 = bbox.y1
        ax.add_patch(
            Rectangle(
                (x0, y0),
                width=x1 - x0,
                height=y1 - y0,
                transform=ax.transAxes,
                color="dimgray",
                alpha=0.5,
            )
        )
        try:
            filename = Path.joinpath(
                raw_file_path.parent, "fft_" + raw_file_path.name
            ).with_suffix(".png")
        except Exception as e:
            print("filenameException", e)
        finally:
            fig.savefig(filename)
            plt.close()
        return

    def process_data_tran(self, raw_file, tracestoplot):
        """
        This is the function that will process the data from transient analysis simulations
        """
        raw_file_path = Path(raw_file)
        raw = RawRead(raw_filename=raw_file_path, traces_to_read="*")

        steps = raw.get_steps()
        ax: Axes
        fig, ax = plt.subplots(
            figsize=(
                self.config["image"]["width"] / self.config["image"]["dpi"],
                self.config["image"]["heigth"] / self.config["image"]["dpi"],
            ),
            dpi=self.config["image"]["dpi"],
            layout="tight",
        )
        ax.set_facecolor(color="black")
        plt.title("Step: " + self.step["name"])

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Vout (V)")
        ax.tick_params(axis="y", which="both")
        ax.grid(which="both", color="white")

        try:
            for i, step in enumerate(steps):
                for j, trace in enumerate(tracestoplot):
                    ax.plot(
                        raw.get_axis(step=step),
                        raw.get_trace(trace).get_wave(step=step),
                        color=self.colors[i * j + j],
                        lw=2,
                        label=f"""{self.step["name"]} - {trace}""",
                    )
        except Exception as e:
            print(e)
            raise
        yabs_max = abs(max(ax.get_ylim(), key=abs))
        ax.set_ylim(
            ymin=self.step.get("y_min", -yabs_max),
            ymax=self.step.get("y_max", yabs_max),
        )
        ax.legend(loc="upper left")

        text_lines = []
        for param in raw_file_path.stem.split("-")[1:]:
            param_name = "_".join(param.split("_")[:-1])
            try:
                param_value = int(param.split("_")[-1])
            except ValueError:
                try:
                    param_value = float(param.split("_")[-1])
                except ValueError:
                    param_value = param.split("_")[-1]
            text_lines.append(f"""{param_name} = {param_value}""")
        text = "\n".join(text_lines)

        pt = ax.text(0, 0, text, fontsize=14, color="white", transform=ax.transAxes)
        r = fig.canvas.get_renderer()
        fig.canvas.draw()
        bbox = pt.get_window_extent(renderer=r).transformed(ax.transAxes.inverted())
        x0 = bbox.x0
        y0 = bbox.y0
        x1 = bbox.x1
        y1 = bbox.y1
        ax.add_patch(
            Rectangle(
                (x0, y0),
                width=x1 - x0,
                height=y1 - y0,
                transform=ax.transAxes,
                color="dimgray",
                alpha=0.5,
            )
        )
        try:
            filename = Path.joinpath(
                raw_file_path.parent, raw_file_path.name
            ).with_suffix(".png")
        except Exception as e:
            print("filenameException", e)
        finally:
            fig.savefig(filename)
            plt.close()
        return

    def process_data(self, raw_file, log_file, step):
        """
        This is the function that will process the data from simulations
        """
        raw_file_path = Path(raw_file)
        try:
            raw = RawRead(raw_filename=raw_file_path, traces_to_read="*")
        except SpiceReadException as e:
            self.logprint(e)
            exit(1)

        raw_trace_names = raw.get_trace_names()
        raw_property = raw.get_raw_property()
        if isinstance(step["tracestoplot"], list):
            wanted_trace_names = step["tracestoplot"]
        else:
            wanted_trace_names = [step["tracestoplot"]]

        trace_names = []
        for trace_name in wanted_trace_names:
            if trace_name in raw_trace_names:
                trace_names.append(trace_name)
            else:
                self.logprint(f"No such trace name: {trace_name} [{raw_trace_names}]")

        if raw_property["Plotname"] == "Transient Analysis":
            self.data_to_analyze.append(
                ("t", raw_file, trace_names, step.get("fft", False))
            )
        elif raw_property["Plotname"] == "AC Analysis":
            self.data_to_analyze.append(("a", raw_file, trace_names, False))

    @staticmethod
    def chunker(list_to_chunk, size):
        """split a list into chunks with length = size"""
        return (
            list_to_chunk[pos : pos + size]
            for pos in range(0, len(list_to_chunk), size)
        )

    def plot_data(self):
        fft_steps_nb = 0
        # matplotlib is not thread safe, we must run separated processes
        chunk_size = self.parallel_plot**2
        if chunk_size > 128:
            chunk_size = 128
        for chunk in self.chunker(self.data_to_analyze, chunk_size):
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.parallel_plot,
            ) as executor:
                futures = []
                for data in chunk:
                    if data[0] == "a":
                        futures.append(
                            executor.submit(
                                self.process_data_ac,
                                raw_file=data[1],
                                tracestoplot=data[2],
                            )
                        )
                    if data[0] == "t":
                        futures.append(
                            executor.submit(
                                self.process_data_tran,
                                raw_file=data[1],
                                tracestoplot=data[2],
                            )
                        )
                        if data[3]:
                            futures.append(
                                executor.submit(
                                    self.process_data_tran_fft,
                                    raw_file=data[1],
                                    tracestoplot=data[2],
                                )
                            )
                            fft_steps_nb += 1
                for _ in concurrent.futures.as_completed(futures):
                    self.data_analyzed += 1
                    self.print_sameline(
                        self.data_analyzed, len(self.data_to_analyze) + fft_steps_nb
                    )

    def janitor(self):
        """
        cleanup generated temp files
        """
        for f in glob(f"{self.temp_folder}/*"):
            os.remove(f)

    def run(self, step):
        """
        run simulation batch
        """
        self.parameters = step["parameters"]
        self.logprint(self.parameters)
        self.data_to_analyze = []
        self.data_analyzed = 0
        # Configures the simulator to use and output folder. Also defines the number of parallel simulations
        runner = SimRunner(
            output_folder=self.temp_folder,
            simulator=LTspice,
            parallel_sims=self.parallel_sim,
            timeout=self.runner_timeout,
            verbose=self.debug,
        )

        netlist = SpiceEditor(LTspice.create_netlist(self.input_file))

        parameters_list = []
        parameters_matrix = []
        pk = tuple(self.parameters.keys())
        for p in self.parameters:
            if isinstance(self.parameters[p], (str, int, float)):
                parameters_matrix.append([self.parameters[p]])
            elif isinstance(self.parameters[p], list):
                parameters_matrix.append(self.parameters[p])
            elif isinstance(self.parameters[p], dict):
                parameters_matrix.append(
                    numpy.arange(
                        float(self.parameters[p]["start"]),
                        float(self.parameters[p]["stop"])
                        + float(self.parameters[p]["step"]),
                        float(self.parameters[p]["step"]),
                    )
                )
        for i in self.product(parameters_matrix):
            parameters_list.append({pk[j]: val for j, val in enumerate(i)})
        if "sim_command" in step:
            netlist.remove_Xinstruction(r"^\.(ac|tran)")
            netlist.add_instruction(step["sim_command"])

        # server = TCPServer(host="localhost", port=5000, mainpid=os.getpid())
        # server_thread = threading.Thread(target=server.start, daemon=True)
        # server_thread.start()

        self.logprint(
            f"""Step {step["name"]} - Processing {len(parameters_list)} combinations"""
        )
        self.start = datetime.now()
        self.logprint("Starting")
        already_done = 0
        for parameters_set in parameters_list:
            netlist.set_parameters(**parameters_set)
            # overriding he automatic netlist naming
            netlist_name_items = []
            for k, v in parameters_set.items():
                # print(k,v)
                if isinstance(v, float):
                    v = f"{v:.15f}"
                netlist_name_items.append(f"""{k}_{v}""")
            run_filename_noext = f"""{unidecode(step["name"]).replace(" ", "_")}-{"-".join(netlist_name_items)}"""
            self.imglist.append(f"{run_filename_noext}.png")
            if self.encode_only:
                continue
            if (
                os.access(f"{self.temp_folder}/{run_filename_noext}.raw", os.R_OK)
                and not self.reset
            ):
                self.process_data(
                    f"{self.temp_folder}/{run_filename_noext}.raw", "", step
                )
                already_done += 1
                continue
            # This will launch up to 'parallel_sims' simulations in background before waiting for resources
            runner.run(
                netlist,
                run_filename=f"{run_filename_noext}.net",
                callback=self.process_data,
                callback_args={"step": step},
            )

            self.print_sameline(
                len(runner.completed_tasks) + already_done, len(parameters_list)
            )

        # This will wait for the all the simulations launched before to complete.
        runner.wait_completion()
        sim_end = datetime.now()
        # The timeout counter is reset everytime a simulation is finished.
        self.logprint(
            f"{sim_end} - Simulation finished in {sim_end - self.start}s ({(sim_end - self.start) / len(parameters_list)} by simulation)\nAlready done and skipped: {already_done}"
        )
        if self.encode_only:
            return
        self.logprint("Preparing plots")
        plot_start = datetime.now()
        self.plot_data()
        plot_end = datetime.now()
        self.logprint(f"Preparing plots done in {plot_end - plot_start}")
        runner.cleanup_files()
        # self.server.stop()

        # Sim Statistics
        self.logprint(
            "Successful/Total Simulations: "
            + str(runner.okSim)
            + "/"
            + str(runner.runno)
        )

    def encode_video(self, step, fft=False):
        """
        make a video with generated images
        """
        if not step or not self.imglist:
            self.logprint("step or imglist is empty, giving up")
            return
        if fft and not self.step.get("fft", False):
            self.logprint(f"""fft not enabled for step {step["name"]}""")
            return

        input_path = Path(self.input_file)
        prefix = "fft_" if fft else ""
        stem = step["name"].replace(" ", "_")
        self.output_file = str(
            input_path.parent / f"{input_path.stem}_{prefix}{stem}.mp4"
        )

        self.output_file = self.input_file.replace(
            "asc", step["name"] + f"""{"_fft_" if fft else ""}.mp4"""
        )
        if len(self.imglist) > 0:
            self.imglist += [self.imglist[-1]] * 2
            temp_folder = Path(self.temp_folder)
            imglist = [
                img
                for img in (f"{prefix}{i}" for i in self.imglist)
                if (temp_folder / img).is_file() and img.startswith(f"{prefix}{stem}")
            ]
            duration = str(self.step.get("ffmpeg_framerate", 1 / self.ffmpeg_framerate))
            with open(self.imglist_file, "w") as f:
                f.write(
                    "\n".join(
                        [f"""file {png}\nduration {duration}""" for png in imglist]
                    )
                )

            self.logprint(f"Encoding video - {self.output_file}")

            try:
                ffmpeg_args = self.set_ffmpeg_args(
                    input=self.imglist_file, ffmpeg_hw_accel=self.ffmpeg_hw_accel
                )
                check_output([self.ffmpeg_bin, *ffmpeg_args, self.output_file])
            except subprocess.CalledProcessError as e:
                self.logprint(f"FFmpeg failed: {e.stderr}")
                raise
        else:
            self.logprint("Empty images list, nothing to encode")

        end = datetime.now()
        try:
            self.logprint(f"Finished in {end - self.start}s")
        except TypeError:
            pass

        # save base64 encoded images
        # with open(f"{Path(__file__).parent}/diaporama.html", "r") as f:
        #     html_tmpl = f.read()
        # b64_imglist = []
        # for img in list(set(imglist)):
        #     b64_imglist.append(f"temp/{img}")
        # with open(
        #     self.input_file.replace(
        #         "asc", step["name"] + f"""{"_fft_" if fft else ""}.html"""
        #     ),
        #     "w",
        # ) as f:
        #     f.write(html_tmpl.replace("__IMAGEDATA__", json.dumps(b64_imglist)))

    def server_stop(self):
        self.server.stop()


def signal_handler(signum, frame):
    print("Ctrl-c was pressed. Exiting")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signalnum=signal.SIGINT, handler=signal_handler)

    argparser = argparse.ArgumentParser(description="Run automated LTSpice simulations")
    argparser.add_argument(
        "-c", "--config", type=str, default=None, help=".yml file to use as config"
    )
    argparser.add_argument(
        "-e",
        "--encode-only",
        action="store_true",
        default=None,
        help="encode video from previous images",
    )
    argparser.add_argument(
        "-s",
        "--skip-encode",
        action="store_true",
        default=None,
        help="skip video encoding",
    )
    argparser.add_argument(
        "-k",
        "--keep-images",
        action="store_true",
        default=None,
        help="keep images after simulation",
    )
    argparser.add_argument(
        "--cleanup", action="store_true", default=False, help="Cleanup temp folder"
    )
    argparser.add_argument(
        "-f",
        "--show-freq-domains",
        action="store_true",
        default=False,
        help="Show audio frequency domains in AC analysis",
    )
    argparser.add_argument(
        "--reset",
        action="store_true",
        default=False,
        help="Start over even if a job can be resumed",
    )
    argparser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="Debug output",
    )
    args = argparser.parse_args()

    ltsb = LTSpiceBatch(args=args)
    if args.cleanup:
        ltsb.janitor()
        sys.exit(0)
    ltsb.test_ffmpeg()
    for step in ltsb.config["steps"]:
        step["name"] = "".join(
            [
                i if i in string.ascii_letters + string.digits + "-_" else "_"
                for i in step["name"]
            ]
        )
        ltsb.step = step
        ltsb.run(step)
        if not args.skip_encode:
            ltsb.encode_video(step=step)
        if step.get("fft", False) and not args.skip_encode:
            ltsb.encode_video(step=step, fft=True)
        if not args.keep_images:
            ltsb.janitor()
    ltsb.server_stop()
