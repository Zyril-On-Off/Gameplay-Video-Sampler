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
from PIL import Image
import webrtcvad as webrtcvad  # Using prebuilt wheels for speech detection

# ----------------------------
# GPU Acceleration Settings
# ----------------------------
def get_ffmpeg_params(use_gpu=False):
    if use_gpu:
        return {"codec": "h264_nvenc", "ffmpeg_params": ["-hwaccel", "cuda"]}
    else:
        return {"codec": "libx264", "ffmpeg_params": []}

def safe_write_videofile(clip, output_path, ffmpeg_opts):
    try:
        clip.write_videofile(
            output_path,
            codec=ffmpeg_opts["codec"],
            ffmpeg_params=ffmpeg_opts["ffmpeg_params"],
            verbose=False
        )
    except Exception as e:
        print("GPU-based encoding failed, falling back to CPU. Error:", e)
        fallback_opts = {"codec": "libx264", "ffmpeg_params": []}
        clip.write_videofile(
            output_path,
            codec=fallback_opts["codec"],
            ffmpeg_params=fallback_opts["ffmpeg_params"],
            verbose=False
        )

# ----------------------------
# Video Processing Helper Functions
# ----------------------------
def extract_video_info(filepath):
    try:
        command = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,avg_frame_rate',
            '-show_entries', 'format=duration',
            '-of', 'json',
            filepath
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        info = json.loads(result.stdout)
        duration = float(info['format']['duration'])
        video_stream = info['streams'][0]
        width = video_stream['width']
        height = video_stream['height']
        fps_str = video_stream.get('avg_frame_rate', '0/0')
        try:
            num, den = fps_str.split('/')
            fps = float(num) / float(den) if float(den) != 0 else 0
        except:
            fps = 0
        return {
            'filename': os.path.basename(filepath),
            'extension': os.path.splitext(filepath)[1],
            'duration': duration,
            'resolution': [width, height],
            'fps': fps
        }
    except Exception as e:
        print("Error extracting video info:", e)
        return None

def detect_scenes(video_path, threshold=30.0):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    video_manager.release()
    return [(scene[0].get_seconds(), scene[1].get_seconds()) for scene in scene_list]

def get_audio_rms(clip):
    if clip.audio is None:
        return 0
    try:
        audio_data = clip.audio.to_soundarray(fps=44100, nbytes=2, quantize=True, buffersize=200000)
        audio_array = np.array(list(audio_data))
        if audio_array.size == 0:
            return 0
        rms = float(np.sqrt(np.mean(audio_array**2)))
        return rms
    except Exception as e:
        print("Error reading audio data:", e)
        return 0

def calculate_motion(clip, sample_rate=1.0):
    num_frames = int(clip.duration // sample_rate)
    if num_frames < 2:
        return 0
    prev_frame = None
    motion_values = []
    for i in range(num_frames):
        t = i * sample_rate
        try:
            frame = clip.get_frame(t)
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                mse = (diff.astype(np.float32)**2).mean()
                motion_values.append(mse)
            prev_frame = gray
        except Exception as e:
            print(f"Error in motion calculation at {t}s: {e}")
    if not motion_values:
        return 0
    return float(np.mean(motion_values))

def get_speech_ratio(clip, sample_rate=16000, frame_duration_ms=30):
    if clip.audio is None:
        return 0
    try:
        audio_data = clip.audio.to_soundarray(fps=sample_rate, nbytes=2, quantize=True, buffersize=200000)
        audio_list = list(audio_data)
        audio_array = np.array(audio_list)
        if audio_array.ndim == 2:
            audio_array = np.mean(audio_array, axis=1)
        audio_array = (audio_array * 32767).astype(np.int16)
        vad = webrtcvad.Vad(3)
        frame_length = int(sample_rate * frame_duration_ms / 1000)
        num_frames = len(audio_array) // frame_length
        speech_frames = 0
        for i in range(num_frames):
            frame = audio_array[i*frame_length:(i+1)*frame_length]
            if len(frame) < frame_length:
                break
            frame_bytes = frame.tobytes()
            if vad.is_speech(frame_bytes, sample_rate):
                speech_frames += 1
        ratio = speech_frames / num_frames if num_frames > 0 else 0
        return ratio
    except Exception as e:
        print("Error in speech detection:", e)
        return 0

def is_scene_interesting(clip, mode="remove_boring", audio_threshold=0.01, motion_threshold=10.0, speech_threshold=0.1):
    audio_value = get_audio_rms(clip)
    audio_interesting = audio_value > audio_threshold
    speech_interesting = False
    motion_score = 0.0
    if mode == "gameplay":
        speech_ratio = get_speech_ratio(clip, sample_rate=16000, frame_duration_ms=30)
        speech_interesting = (speech_ratio >= speech_threshold)
        motion_score = calculate_motion(clip, sample_rate=1.0)
    if mode == "remove_boring":
        return audio_interesting
    elif mode == "gameplay":
        return (motion_score >= motion_threshold) and (speech_interesting or audio_interesting)
    elif mode == "moments":
        return audio_interesting
    else:
        return audio_interesting

def split_and_export(final_clip, output_video, segment_length=0, ffmpeg_opts=None):
    if ffmpeg_opts is None:
        ffmpeg_opts = get_ffmpeg_params(use_gpu=False)
    if segment_length <= 0 or final_clip.duration <= segment_length:
        safe_write_videofile(final_clip, output_video, ffmpeg_opts)
    else:
        num_segments = int(final_clip.duration // segment_length)
        if final_clip.duration % segment_length:
            num_segments += 1
        for i in range(num_segments):
            start_time = i * segment_length
            end_time = min((i+1) * segment_length, final_clip.duration)
            segment_clip = final_clip.subclip(start_time, end_time)
            segment_output = output_video.replace(".mp4", f"_part{i+1}.mp4")
            safe_write_videofile(segment_clip, segment_output, ffmpeg_opts)

def export_automatic(clips, output_video, ffmpeg_opts):
    final_clip = concatenate_videoclips(clips, method="compose")
    safe_write_videofile(final_clip, output_video, ffmpeg_opts)

def export_multiple_clips(final_clip, output_video, num_clips, clip_length_sec, ffmpeg_opts):
    if num_clips <= 1:
        safe_write_videofile(final_clip, output_video, ffmpeg_opts)
        return
    total_duration = final_clip.duration
    for i in range(num_clips):
        start_time = i * clip_length_sec
        if start_time >= total_duration:
            break
        end_time = min(start_time + clip_length_sec, total_duration)
        segment_clip = final_clip.subclip(start_time, end_time)
        segment_output = output_video.replace(".mp4", f"_clip{i+1}.mp4")
        safe_write_videofile(segment_clip, segment_output, ffmpeg_opts)

# ----------------------------
# Safe Write Function with GPU Fallback
# ----------------------------
def safe_write_videofile(clip, output_path, ffmpeg_opts):
    try:
        clip.write_videofile(
            output_path,
            codec=ffmpeg_opts["codec"],
            ffmpeg_params=ffmpeg_opts["ffmpeg_params"],
            verbose=False
        )
    except Exception as e:
        print("GPU-based encoding failed, falling back to CPU. Error:", e)
        fallback_opts = {"codec": "libx264", "ffmpeg_params": []}
        clip.write_videofile(
            output_path,
            codec=fallback_opts["codec"],
            ffmpeg_params=fallback_opts["ffmpeg_params"],
            verbose=False
        )

# ----------------------------
# Main GUI Application
# ----------------------------
class VideoProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gameplay Video Sampler")
        self.filepath = None
        self.video_duration = 0
        self.thumbnail_img = None
        self.cancel_processing = False
        self.processing = False
        self.create_widgets()
        self.root.bind("<space>", self.on_space_press)
        
    def create_widgets(self):
        self.file_frame = ttk.LabelFrame(self.root, text="Import Video File")
        self.file_frame.pack(fill="both", padx=10, pady=5)
        
        self.drop_label = ttk.Label(self.file_frame, text="Drag your video file here", background="lightgray")
        self.drop_label.pack(expand=True, fill="both", padx=10, pady=5)
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind("<<Drop>>", self.handle_drop)
        
        self.browse_button = ttk.Button(self.file_frame, text="Browse Files", command=self.browse_file)
        self.browse_button.pack(padx=10, pady=5)
        
        self.info_frame = ttk.LabelFrame(self.root, text="File Information")
        self.info_frame.pack(fill="both", padx=10, pady=5)
        
        self.info_text = tk.Text(self.info_frame, height=5, width=50)
        self.info_text.pack(side="left", padx=10, pady=10)
        
        self.thumbnail_label = ttk.Label(self.info_frame)
        self.thumbnail_label.pack(side="left", padx=10, pady=10)
        
        self.timeline_frame = ttk.LabelFrame(self.root, text="Timeline")
        self.timeline_frame.pack(fill="both", padx=10, pady=5)
        self.timeline_canvas = tk.Canvas(self.timeline_frame, width=400, height=50, bg="black")
        self.timeline_canvas.pack(padx=10, pady=5)
        
        self.options_frame = ttk.LabelFrame(self.root, text="Processing Options")
        self.options_frame.pack(fill="both", padx=10, pady=5)
        
        self.mode_var = tk.StringVar(value="remove_boring")
        modes = [
            ("Remove Boring Parts", "remove_boring"),
            ("Gameplay Focus", "gameplay"),
            ("Moments Only", "moments")
        ]
        for text, mode in modes:
            rb = ttk.Radiobutton(self.options_frame, text=text, variable=self.mode_var, value=mode)
            rb.pack(anchor="w", padx=10, pady=2)
        
        self.gpu_var = tk.BooleanVar(value=False)
        self.gpu_check = ttk.Checkbutton(self.options_frame, text="Use GPU Acceleration", variable=self.gpu_var)
        self.gpu_check.pack(anchor="w", padx=10, pady=2)
        
        self.game_type_label = ttk.Label(self.options_frame, text="Game Type:")
        self.game_type_label.pack(anchor="w", padx=10, pady=2)
        self.game_type_var = tk.StringVar(value="Casual")
        self.game_type_option = ttk.Combobox(self.options_frame, textvariable=self.game_type_var, state="readonly")
        self.game_type_option['values'] = ["FPS", "Horror", "MMORPG", "Racing", "Casual"]
        self.game_type_option.pack(anchor="w", padx=10, pady=2)
        
        # Advanced Settings: Debug Mode
        self.advanced_frame = ttk.LabelFrame(self.root, text="Advanced Settings")
        self.advanced_frame.pack(fill="both", padx=10, pady=5)
        self.debug_var = tk.BooleanVar(value=False)
        self.debug_check = ttk.Checkbutton(self.advanced_frame, text="Debug Mode (Export sample scene only)", variable=self.debug_var)
        self.debug_check.pack(anchor="w", padx=10, pady=2)
        
        self.export_frame = ttk.LabelFrame(self.root, text="Export Settings")
        self.export_frame.pack(fill="both", padx=10, pady=5)
        
        self.export_mode_label = ttk.Label(self.export_frame, text="Export Mode:")
        self.export_mode_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.export_mode_var = tk.StringVar(value="Automatic")
        self.export_mode_option = ttk.Combobox(self.export_frame, textvariable=self.export_mode_var, state="readonly")
        self.export_mode_option['values'] = ["Automatic", "Fixed Clip Length", "Multiple Clips"]
        self.export_mode_option.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.export_mode_option.bind("<<ComboboxSelected>>", self.update_export_fields)
        
        self.fixed_length_label = ttk.Label(self.export_frame, text="Max Clip Length (minutes):")
        self.fixed_length_entry = ttk.Entry(self.export_frame, width=5)
        self.fixed_length_entry.insert(0, "0")
        
        self.multiple_num_label = ttk.Label(self.export_frame, text="Number of Clips:")
        self.multiple_num_entry = ttk.Entry(self.export_frame, width=5)
        self.multiple_num_entry.insert(0, "1")
        self.multiple_length_label = ttk.Label(self.export_frame, text="Clip Length (minutes):")
        self.multiple_length_entry = ttk.Entry(self.export_frame, width=5)
        self.multiple_length_entry.insert(0, "0")
        
        self.fixed_length_label.grid_remove()
        self.fixed_length_entry.grid_remove()
        self.multiple_num_label.grid_remove()
        self.multiple_num_entry.grid_remove()
        self.multiple_length_label.grid_remove()
        self.multiple_length_entry.grid_remove()
        
        self.process_button = ttk.Button(self.root, text="Process Video", command=self.start_processing)
        self.process_button.pack(pady=10)
        
        self.status_label = ttk.Label(self.root, text="Status: Waiting for file...")
        self.status_label.pack(pady=5)
        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=5)
        
    def update_export_fields(self, event=None):
        mode = self.export_mode_var.get()
        self.fixed_length_label.grid_remove()
        self.fixed_length_entry.grid_remove()
        self.multiple_num_label.grid_remove()
        self.multiple_num_entry.grid_remove()
        self.multiple_length_label.grid_remove()
        self.multiple_length_entry.grid_remove()
        if mode == "Fixed Clip Length":
            self.fixed_length_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
            self.fixed_length_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        elif mode == "Multiple Clips":
            self.multiple_num_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
            self.multiple_num_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
            self.multiple_length_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
            self.multiple_length_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
    def on_space_press(self, event):
        if self.processing:
            if messagebox.askyesno("Cancel Processing", "Processing is in progress. Do you want to cancel it?"):
                self.cancel_processing = True
                self.update_progress(0, "Processing cancelled by user.")
        
    def update_progress(self, value, message=""):
        self.progress['value'] = value
        self.status_label.config(text=f"Status: {message}")
        self.root.update_idletasks()
        
    def handle_drop(self, event):
        self.filepath = event.data.strip('{}')
        self.update_file_info()
        
    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.mov *.avi *.mkv")]
        )
        if filename:
            self.filepath = filename
            self.update_file_info()
        
    def update_file_info(self):
        if self.filepath and os.path.isfile(self.filepath):
            info = extract_video_info(self.filepath)
            if info:
                self.video_duration = info['duration']
                info_str = (
                    f"Filename: {info['filename']}\n"
                    f"Extension: {info['extension']}\n"
                    f"Duration: {info['duration']:.2f} sec\n"
                    f"Resolution: {info['resolution'][0]}x{info['resolution'][1]}\n"
                    f"FPS: {info['fps']}\n"
                )
                self.info_text.delete("1.0", tk.END)
                self.info_text.insert(tk.END, info_str)
                self.status_label.config(text="File loaded. Ready to process.")
                self.load_thumbnail()
            else:
                self.info_text.delete("1.0", tk.END)
                self.info_text.insert(tk.END, "Error reading file information.")
        else:
            self.info_text.delete("1.0", tk.END)
            self.info_text.insert(tk.END, "No valid file loaded.")
            
    def load_thumbnail(self):
        try:
            clip = VideoFileClip(self.filepath)
            t = 1 if clip.duration > 1 else clip.duration / 2
            frame = clip.get_frame(t)
            clip.close()
            image = Image.fromarray(frame.astype('uint8'), 'RGB')
            image.thumbnail((200, 200))
            self.thumbnail_img = ImageTk.PhotoImage(image)
            self.thumbnail_label.config(image=self.thumbnail_img)
        except Exception as e:
            print("Error loading thumbnail:", e)
        
    def draw_timeline(self, scenes, interesting_flags, total_duration):
        canvas_width = 400
        canvas_height = 50
        self.timeline_canvas.delete("all")
        for i, (start, end) in enumerate(scenes):
            x1 = (start / total_duration) * canvas_width
            x2 = (end / total_duration) * canvas_width
            color = "green" if interesting_flags[i] else "red"
            self.timeline_canvas.create_rectangle(x1, 0, x2, canvas_height, fill=color)
        
    def start_processing(self):
        if not self.filepath:
            messagebox.showerror("Error", "Please load a video file first.")
            return
        
        if self.processing:
            if messagebox.askyesno("Cancel Processing", "Processing is already in progress. Do you want to cancel it?"):
                self.cancel_processing = True
            return
        
        self.cancel_processing = False
        self.processing = True
        self.update_progress(5, "Starting processing...")
        threading.Thread(target=self.process_video).start()
        
    def process_video(self):
        mode = self.mode_var.get()
        export_mode = self.export_mode_var.get()  # "Automatic", "Fixed Clip Length", or "Multiple Clips"
        use_gpu = self.gpu_var.get()
        ffmpeg_opts = get_ffmpeg_params(use_gpu)
        
        game_type = self.game_type_var.get()
        game_type_thresholds = {
            "FPS": 0.15,
            "Horror": 0.05,
            "MMORPG": 0.1,
            "Racing": 0.1,
            "Casual": 0.1
        }
        speech_threshold = game_type_thresholds.get(game_type, 0.1)
        
        output_video = filedialog.asksaveasfilename(
            title="Save Processed Video As",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4")]
        )
        if not output_video:
            self.update_progress(0, "Processing cancelled: No output file selected.")
            self.processing = False
            return

        try:
            # Debug Mode: if enabled, bypass full scene detection and use a sample scene
            if self.debug_var.get():
                self.update_progress(30, "Debug Mode: Using sample scene (first 30 seconds).")
                video = VideoFileClip(self.filepath)
                selected_clips = [video.subclip(0, min(30, self.video_duration))]
            else:
                self.update_progress(10, "Detecting scenes...")
                scenes = detect_scenes(self.filepath, threshold=30.0)
                self.update_progress(20, f"Scene detection complete: {len(scenes)} scenes found.")
                if not scenes:
                    self.update_progress(0, "No scenes detected.")
                    self.processing = False
                    return
                
                interesting_flags = [False] * len(scenes)
                self.draw_timeline(scenes, interesting_flags, self.video_duration)
                
                video = VideoFileClip(self.filepath)
                selected_clips = []
                total_scenes = len(scenes)
                
                for i, (start, end) in enumerate(scenes):
                    if self.cancel_processing:
                        self.update_progress(0, "Processing cancelled by user.")
                        video.close()
                        self.processing = False
                        return
                    clip = video.subclip(start, end)
                    interesting = False
                    if mode == "remove_boring":
                        interesting = get_audio_rms(clip) > 0.01
                    elif mode == "gameplay":
                        motion = calculate_motion(clip, sample_rate=1.0)
                        speech_ratio = get_speech_ratio(clip, sample_rate=16000, frame_duration_ms=30)
                        interesting = (motion >= 10.0) and (speech_ratio >= speech_threshold or get_audio_rms(clip) > 0.01)
                    elif mode == "moments":
                        interesting = get_audio_rms(clip) > 0.01
                    interesting_flags[i] = interesting
                    if interesting:
                        selected_clips.append(clip)
                    progress_value = 20 + ((i + 1) / total_scenes) * 40
                    self.update_progress(progress_value, f"Processing scene {i+1} of {total_scenes}...")
                    self.draw_timeline(scenes, interesting_flags, self.video_duration)
            
            if not selected_clips:
                self.update_progress(100, "No interesting scenes found based on the selected criteria.")
                video.close()
                self.processing = False
                return
            
            self.update_progress(70, "Exporting final video...")
            final_clip = concatenate_videoclips(selected_clips, method="compose")
            
            if export_mode == "Automatic":
                safe_write_videofile(final_clip, output_video, ffmpeg_opts)
            elif export_mode == "Fixed Clip Length":
                try:
                    fixed_length_min = float(self.fixed_length_entry.get())
                except ValueError:
                    fixed_length_min = 0
                clip_length_sec = fixed_length_min * 60
                split_and_export(final_clip, output_video, segment_length=clip_length_sec, ffmpeg_opts=ffmpeg_opts)
            elif export_mode == "Multiple Clips":
                try:
                    num_clips = int(self.multiple_num_entry.get())
                except ValueError:
                    num_clips = 1
                try:
                    clip_length_min = float(self.multiple_length_entry.get())
                except ValueError:
                    clip_length_min = 0
                clip_length_sec = clip_length_min * 60
                export_multiple_clips(final_clip, output_video, num_clips, clip_length_sec, ffmpeg_opts)
            
            self.update_progress(100, f"Processing complete. Output saved at or near {output_video}")
            video.close()
            self.processing = False
        except Exception as e:
            self.update_progress(0, f"Error during processing: {e}")
            self.processing = False

    def update_progress(self, value, message=""):
        self.progress['value'] = value
        self.status_label.config(text=f"Status: {message}")
        self.root.update_idletasks()

    def handle_drop(self, event):
        self.filepath = event.data.strip('{}')
        self.update_file_info()

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.mov *.avi *.mkv")]
        )
        if filename:
            self.filepath = filename
            self.update_file_info()

    def update_file_info(self):
        if self.filepath and os.path.isfile(self.filepath):
            info = extract_video_info(self.filepath)
            if info:
                self.video_duration = info['duration']
                info_str = (
                    f"Filename: {info['filename']}\n"
                    f"Extension: {info['extension']}\n"
                    f"Duration: {info['duration']:.2f} sec\n"
                    f"Resolution: {info['resolution'][0]}x{info['resolution'][1]}\n"
                    f"FPS: {info['fps']}\n"
                )
                self.info_text.delete("1.0", tk.END)
                self.info_text.insert(tk.END, info_str)
                self.status_label.config(text="File loaded. Ready to process.")
                self.load_thumbnail()
            else:
                self.info_text.delete("1.0", tk.END)
                self.info_text.insert(tk.END, "Error reading file information.")
        else:
            self.info_text.delete("1.0", tk.END)
            self.info_text.insert(tk.END, "No valid file loaded.")

    def load_thumbnail(self):
        try:
            clip = VideoFileClip(self.filepath)
            t = 1 if clip.duration > 1 else clip.duration / 2
            frame = clip.get_frame(t)
            clip.close()
            image = Image.fromarray(frame.astype('uint8'), 'RGB')
            image.thumbnail((200, 200))
            self.thumbnail_img = ImageTk.PhotoImage(image)
            self.thumbnail_label.config(image=self.thumbnail_img)
        except Exception as e:
            print("Error loading thumbnail:", e)

    def draw_timeline(self, scenes, interesting_flags, total_duration):
        canvas_width = 400
        canvas_height = 50
        self.timeline_canvas.delete("all")
        for i, (start, end) in enumerate(scenes):
            x1 = (start / total_duration) * canvas_width
            x2 = (end / total_duration) * canvas_width
            color = "green" if interesting_flags[i] else "red"
            self.timeline_canvas.create_rectangle(x1, 0, x2, canvas_height, fill=color)

    def on_space_press(self, event):
        if self.processing:
            if messagebox.askyesno("Cancel Processing", "Processing is in progress. Do you want to cancel it?"):
                self.cancel_processing = True
                self.update_progress(0, "Processing cancelled by user.")

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = VideoProcessorGUI(root)
    root.mainloop()
