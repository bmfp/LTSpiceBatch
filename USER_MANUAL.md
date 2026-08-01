# LTSpiceBatch User Manual

This manual explains how to run LTSpiceBatch from zero to finished simulation images/videos.

## 1. Before You Start

You need:

- LTSpice installed (required).
- FFmpeg installed and accessible by path (included).
- Python dependencies installed (just let `uv` do it for you).
- A valid LTSpice schematic (`.asc`) with parameters and traces you want to sweep.

## 2. Launch the Application

### Windows

Run:

```powershell
./run-sim.bat
```

### Linux

Run:

```sh
./run-sim.sh
```

This opens the GUI (`launcher.py`).

## 3. Configure Global Settings

In the "Global Config" tab:

1. Set `input_file` to your LTSpice `.asc` file.
2. Optionally tune:
   - `ffmpeg_bin`
   - `ffmpeg_framerate`
   - `image` resolution and DPI
   - `parallel_sim`
   - `parallel_plot`
   - `runner_timeout`
   - `temp_folder`

Important:

- Use a writable `temp_folder` if you override it.

## 4. Create Simulation Steps

Go to "Steps" and click "Add Step".

Each step defines one simulation scenario.

Required fields:

- `name`
- `sim_command` (example: `.ac dec 100 20 20k`, `.tran {10/freq}`)
- `tracestoplot` (one or more LTSpice traces)

Optional fields:

- `ffmpeg_framerate` per step
- `fft` for transient FFT chart generation
- `fft_x_max`
- y-axis bounds (`y_min`, `y_max`, `mag_y_min`, `mag_y_max`, `phase_y_min`, `phase_y_max`)

## 5. Define Parameter Sweeps

In each step, add parameters with one of three types:

1. Single value
2. List of values
3. Range (`start`, `stop`, `step`)

Example:

```yaml
parameters:
  c2: 6.8n
  freq: [10, 73, 259, 657, "1k"]
  c1:
    start: 1e-12
    stop: 1e-10
    step: 2e-12
```

How combinations are built:

- LTSpiceBatch computes the Cartesian product of all parameter sets.
- Total simulations per step = product of each parameter count.

## 6. Save and Run

In "Save and Run":

1. Click "Save YAML" (or "Save and Run" directly).
2. Choose a `.yml` or `.yaml` output file.
3. Monitor run logs in the output panel.
4. Use "Stop" to request graceful interruption.

## 7. Command-Line Mode

You can run without the GUI:

```sh
uv run LTSpiceBatch.py -c path/to/config.yml
```

Useful flags:

- `--encode-only`: skip simulation and only build videos from existing images.
- `--skip-encode`: run simulation and plotting only.
- `--keep-images`: preserve generated frames and copy to timestamped folder.
- `--reset`: force rerun even if resumable `.raw` exists.
- `--cleanup`: clear temp outputs.
- `--show-freq-domains`: show AC band overlays.

## 8. Understand Output Files

Generated image files (temp folder):

- `<step>-<param_set>.png`
- `fft_<step>-<param_set>.png` if FFT is enabled.

Generated videos (near input `.asc`):

- `<input>.<step>.mp4`
- `<input>.<step>_fft_.mp4` if FFT was encoded.

If `--keep-images` is used:

- Images are copied to a timestamp folder next to the input file.

## 9. Review Plots with the Slideshow Tool

Launch:

### Windows

```powershell
./run-show.bat
```

### Linux

```sh
./run-show.sh
```

Then:

1. Open a folder containing generated PNG files (and optionally `*_imglist.txt`, recommended to preserve order).
2. Browse with controls or mouse wheel.
3. Filter using metadata extracted from PNG `Description` fields.

## 10. Troubleshooting

### FFmpeg not found or not executable

- Set an explicit absolute path in `ffmpeg_bin`.
- Verify execute permissions.

### No plots or missing traces

- Ensure each `tracestoplot` item exactly matches LTSpice trace names in `.raw`.

### Simulations appear skipped unexpectedly

- Existing `.raw` files are reused unless `--reset` is set.

### Linux + LTSpice compatibility

- Configure Wine values in launcher optional fields:
  - `wine_executable`
  - `wine_folder`

### Long jobs or freeze concerns

- Reduce `parallel_sim` and `parallel_plot`.
- Increase `runner_timeout` for slow circuits.

## 11. Recommended Workflow

1. Start from the included sample files.
2. Validate one small step first.
3. Expand parameter sweeps gradually.
4. Enable FFT only where useful.
5. Keep images during tuning, then disable for final runs (if you just want a video).
