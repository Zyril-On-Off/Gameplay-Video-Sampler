# Gameplay Video Sampler

**Gameplay Video Sampler** is a tool designed to process and edit gameplay footage automatically. It was created as a fun project to streamline the editing process and make it easier to share exciting gameplay moments with friends.

## Description
Just created this as a fun project since we had so many gameplay clips with my friends. I got interested in editing, but I am too lazy. Instead, I just created a system that will edit for me—that's why I created this. It's not that good; I just stopped developing it until I reached the video result that I wanted, covering the removal of boring parts, gameplay focus, and moments only.

Some of the choices are redundant, so maybe continuing with trial and error is better.

## Features
- **Smart Sampling:** Automatically removes boring parts of the footage.
- **Highlight Compilation:** Focuses on gameplay highlights and significant moments.
- **Customization Options:** 
  - Remove Boring Parts
  - Gameplay Focus
  - Moments Only
- **Advanced Settings:**
  - **GPU Acceleration (In Testing):** Uses GPU for faster processing, but may fail sometimes.
  - **CPU Processing:** Fallback to CPU if GPU fails.
  - **Debug Mode:** Exports a sample of the video to test if the system works on your computer.

## Export Modes
- **Automatic:** Compiles all highlights into one clip. For example, a 1-hour clip might be cut down to 20-30 minutes.
- **Fixed Clip Length:** Set the length of each clip. If there are more interesting clips, they will be exported in multiple parts.
- **Controlled Fixed Length:** Set a specific length and amount for more controlled exports.

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
   ```bash
   git clone https://github.com/Zyril-On-Off/Gameplay-Video-Sampler.git
   cd Gameplay-Video-Sampler
