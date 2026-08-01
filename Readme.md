# LTSpiceBatch

LTSpiceBatch automates LTSpice simulation sweeps and converts results into plots and MP4 videos.

This folder provides:

- A GUI config launcher (`launcher.py`) to create/edit YAML job files and run jobs.
- A batch engine (`LTSpiceBatch.py`) to execute LTSpice simulations, generate plots, and encode videos.
- A slideshow viewer (`diapo.py`) to inspect generated PNG frames and metadata.

## What It Does

For each step in a YAML config, LTSpiceBatch can:

1. Expand parameter combinations (single values, lists, ranges).
2. Run LTSpice simulations in parallel.
3. Parse `.raw` outputs via PyLTSpice.
4. Render plots (`.png`) for AC and transient analyses.
5. Optionally generate FFT plots for transient runs.
6. Encode images into MP4 via FFmpeg.

## Project Layout

- `launcher.py`: Tkinter UI to build YAML config and run jobs.
- `LTSpiceBatch.py`: command-line batch processor.
- `diapo.py`: image slideshow browser for generated PNG output.
- `high-low pass.asc`: LTSpice sample schematic.
- `high-low pass.yml`: sample YAML job.
- `run-sim.bat` / `run-sim.sh`: launch GUI quickly.
- `run-show.bat` / `run-show.sh`: launch slideshow viewer quickly.

## Requirements

- LTSpice installed and runnable (standard install on Windows, use wine appimage on linux).
- FFmpeg executable (included).
- Python dependencies from `pyproject.toml`:
  - `pyltspice`
  - `pyyaml`
  - `chardet`
  - `unidecode`
- Optional but recommended: `uv` (included, used by helper scripts and launcher execution path).

## Quick Start

### Windows

1. Open a terminal in this folder.
2. Start the GUI:

```powershell
./run-sim.bat
```

3. In the GUI:
   - Set `input_file` to your `.asc` schematic.
   - Add one or more steps.
   - Save YAML.
   - Click "Save and Run".

### Linux

1. Open a terminal in this folder.
2. Start the GUI:

```sh
./run-sim.sh
```

## CLI Usage

Run the engine directly:

```sh
uv run LTSpiceBatch.py -c path/to/config.yml
```

Command options:

- `-c, --config`: path to YAML config file.
- `-e, --encode-only`: only encode from existing generated images.
- `-s, --skip-encode`: run simulations/plots but skip MP4 encoding.
- `-k, --keep-images`: keep generated images after run and copy to timestamped output directory.
- `--cleanup`: clean temporary folder before/while running.
- `-f, --show-freq-domains`: overlay audio frequency bands in AC plots.
- `--reset`: re-run even if resumable artifacts already exist.
- `-d, --debug`: verbose output.

## YAML Configuration

Top-level keys:

- `input_file` (required): LTSpice `.asc` file path.
- `steps` (required): list of simulation steps.
- `ffmpeg_bin` (optional): path to FFmpeg executable.
- `ffmpeg_framerate` (optional): default video framerate.
- `ffmpeg_hw_accel` (optional): force hardware encoding behavior.
- `image` (optional): `{ dpi, width, heigth }`.
- `parallel_sim` (optional): number of parallel simulation jobs.
- `parallel_plot` (optional): number of parallel plotting workers.
- `runner_timeout` (optional): simulation timeout in seconds.
- `temp_folder` (optional): temp working directory.
- `debug` (optional): debug mode.

Each step supports:

- `name`: output name token.
- `sim_command`: command like `.ac ...` or `.tran ...`.
- `tracestoplot`: one trace or list of traces.
- `parameters`: map of parameter sweeps.
- `fft` (optional, boolean): enable FFT output for transient step.
- `fft_x_max` (optional): max x-axis for FFT chart.
- `ffmpeg_framerate` (optional): override global framerate for this step.
- `y_min` / `y_max` (optional): transient y limits.
- `mag_y_min` / `mag_y_max` (optional): AC magnitude limits.
- `phase_y_min` / `phase_y_max` (optional): AC phase limits.

Parameter value types:

- Single value: `c1: 6.8e-9`
- List: `freq: [10, 73, 259]`
- Range:

```yaml
freq:
  start: 20
  stop: 20000
  step: 20
```

## Example Config

```yaml
---
ffmpeg_framerate: 6
image:
  dpi: 100
  width: 1920
  heigth: 1080
input_file: ./high-low pass.asc
parallel_sim: 4
parallel_plot: 4
runner_timeout: 30
steps:
  - name: High-pass AC
    sim_command: .ac dec 100 20 20k
    tracestoplot:
      - V(out_passe_haut)
      - V(out_passe_bas)
    parameters:
      c1:
        start: 1e-12
        stop: 1e-10
        step: 2e-12
      c2: 6.8e-9
```

You can use any unit suffix as you would in LTSpice, such as meg, k, p, ...

## Outputs

- Generated plots in temp folder:
  - `<step>-<params>.png`
  - `fft_<step>-<params>.png` (if FFT enabled)
- Encoded videos in input schematic folder:
  - `<input>.<step>.mp4`
  - `<input>.<step>_fft_.mp4` (if FFT enabled)

## Notes

- On Linux, launcher can pass `WINEEXECUTABLE` and `WINEFOLDER` environment variables (useful if you choose wine appimage).
- The batch process can be stopped from launcher via local TCP control (`localhost:5000`).

## Related Documents

- `USER_MANUAL.md`: task-oriented user guide in English.
- `USER_MANUAL.fr.md`: guide utilisateur complet (français).
- `config.template.yml`: minimal starter configuration.