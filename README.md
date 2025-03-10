# Gameplay Video Sampler

`<div align="center">
  <img src="https://i.imgur.com/YEJ9BUF.png" alt="System Interface" width="600">
  <br><br>
  <p>
    <a href="https://www.twitch.tv/gameakawnt">
      <img src="https://static.twitchcdn.net/assets/favicon-32-d6025c14e900565d6177.png" alt="Twitch Logo" width="24" height="24" style="vertical-align: middle;">
      Follow on Twitch: gameakawnt
    </a>
  </p>
  <p>
    <img src="https://www.python.org/static/community_logos/python-logo-generic.svg" alt="Python Logo" width="200">
  </p>
</div>`## Overview

Gameplay Video Sampler is a tool designed to process and edit gameplay footage automatically. It was created as a fun project to streamline the editing process and make it easier to share exciting gameplay moments with friends.

## Description

Just created this as a fun project since we had so many gameplay clips with my friends. I got interested in editing, but I am too lazy. Instead, I just created a system that will edit for me—that's why I created this. It's not that good; I just stopped developing it until I reached the video result that I wanted, covering the removal of boring parts, gameplay focus, and moments only.

Some of the choices are redundant, so maybe continuing with trial and error is better.

## Features

### Smart Sampling

- Automatically removes boring parts of the footage.


### Highlight Compilation

- Focuses on gameplay highlights and significant moments.


### Customization Options

- Remove Boring Parts
- Gameplay Focus
- Moments Only


### Advanced Settings

- **GPU Acceleration** (In Testing): Uses GPU for faster processing, but may fail sometimes.
- **CPU Processing**: Fallback to CPU if GPU fails.
- **Debug Mode**: Exports a sample of the video to test if the system works on your computer.


## Export Modes

- **Automatic**: Compiles all highlights into one clip. For example, a 1-hour clip might be cut down to 20-30 minutes.
- **Fixed Clip Length**: Set the length of each clip. If there are more interesting clips, they will be exported in multiple parts.
- **Controlled Fixed Length**: Set a specific length and amount for more controlled exports.


## Supported Game Types

This tool covers most games we play, including:

- FPS
- Horror
- MMORPG
- Racing
- Casual


## Timeline

The timeline feature ensures the system is running correctly.

## Media Extensions

The system may not work on all media extensions.

## Getting Started

### Prerequisites

- Python 3.x
- Virtual Environment (venv)


### Setup

1. **Clone the Repository**

```shellscript
git clone https://github.com/Zyril-On-Off/Gameplay-Video-Sampler.git
cd Gameplay-Video-Sampler
```


2. **Create a Virtual Environment**

```shellscript
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
```


3. **Install Requirements**

```shellscript
pip install -r requirements.txt
```




### Important Imports

The following are the necessary imports for this project:

```python
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
import cv2
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import VideoFileClip, concatenate_videoclips
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from PIL import Image, ImageTk
import subprocess
import json
import torch
from torchvision import transforms
import torch.nn as nn
import webrtcvad as webrtcvad
```

## Tutorial

1. **Import Video**: Upload your gameplay footage.
2. **Select Options**: Choose your desired settings for sampling and compilation.
3. **Run Debug**: Use the debug mode to test if the system works on your computer.
4. **Export Video**: Choose an export mode and get your edited gameplay video.


## Contributing

If you'd like to contribute, please ask for permission first by sending a message on my Discord: [Discord Invite](https://discord.gg/gKQeW4mfwf)

## License

This project is licensed under the MIT License.

## Visual Overview

`<div align="center">
  <img src="https://i.imgur.com/YEJ9B`