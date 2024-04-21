
# LTSpiceBatch

This tool aims at helping you to easily batch run and visualize LTSpice simulations : configure your job in a few yaml lines and watch a video moments later

  

## Installation (Windows)

- Download ffmpeg
https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-essentials.7z
- Extract it where convenient
- Download python (embeddable package 64bits)
https://www.python.org/downloads/windows/
- Extract it where convenient
- Download pip
https://bootstrap.pypa.io/get-pip.py
- Install pip
```
cd c:\path\to\extracted\python\
.\python.exe c:\path\to\get-pip.py
```
- Modify python3\<your version\>._pth
add `Lib\site-packages` on first blank line
- Install dependancies
```
cd c:\path\to\extracted\python\
.\Scripts\pip.exe install pyltspice pyyaml
```

## Usage
- Configure your job as shown in example file
- Run it
```
c:\path\to\extracted\python\python.exe c:\path\to\LTSpiceBatch.py -c 'c:\path\to\\yourconfig.yml' -h
usage: LTSpiceBatch.py [-h] [-c CONFIG] [-e] [-k] [--cleanup] [-f] [--reset]

Run automated LTSpice simulations

  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        .yml file to use as config
  -e, --encode-only     encode video from previous images
  -k, --keep-images     keep images after simulation
  --cleanup             Cleanup temp folder
  -f, --show-freq-domains
                        Show frequence domains in AC analysis
  --reset               Start over even if a job can be resumed
```
- Check your results as mp4 video file