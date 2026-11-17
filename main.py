import os
import sys
import re
import shutil
import subprocess
import threading
import gc
import multiprocessing
import customtkinter as ctk
from tkinter import filedialog, messagebox
from pathlib import Path
import pysrt

# --- HACK: Prevent transformers from crashing due to broken vision libraries in Windows ---
sys.modules['torchvision'] = None

# Auto-Detect CUDA safely AND fix DLL path for Faster-Whisper
try:
    import torch
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
    os.environ["PATH"] = torch_lib_path + os.pathsep + os.environ.get("PATH", "")
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

# Setup general appearance (Dark & Modern UI)
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# ==========================================
# 1. THE BACKEND ENGINE (PRO PIPELINE)
# ==========================================
class AudioSeparatorEngine:
    """Pro-grade Backend Separation, Transcription & Translation Engine"""
    def __init__(self, output_directory="Output_Files"):
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.lang_codes = {
            "Arabic": "arb_Arab", 
            "English": "eng_Latn",
            "French": "fra_Latn",
            "Spanish": "spa_Latn",
            "German": "deu_Latn"
        }
        
        self.whisper_to_nllb = {
            "en": "eng_Latn", "ar": "arb_Arab", "fr": "fra_Latn",
            "es": "spa_Latn", "de": "deu_Latn", "ru": "rus_Cyrl",
            "it": "ita_Latn", "ja": "jpn_Jpan", "ko": "kor_Hang",
            "zh": "zho_Hans", "tr": "tur_Latn", "pt": "por_Latn"
        }

    def process_pipeline(self, file_path, mode, target_lang, model_size, hardware_type, gpu_index, compute_type, logger_callback):
        input_path = Path(file_path)
        
        if not input_path.exists():
            logger_callback("❌ Error: File not found!")
            return False

        # --- FIX: تنظيف وتقصير اسم الملف لتجنب حد 260 حرف في ويندوز ---
        clean_stem = re.sub(r'[^\w\s-]', '', input_path.stem).strip()
        clean_stem = re.sub(r'[-\s]+', '_', clean_stem)[:40]
        if not clean_stem:
            clean_stem = "processed_media"

        logger_callback(f"⏳ Starting Professional Pipeline for: {input_path.name}...")
        
        env = os.environ.copy()
        ffmpeg_exe = "ffmpeg"
        env["PYTHONIOENCODING"] = "utf-8"
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0

        result_folder = self.output_dir / "htdemucs" / clean_stem
        result_folder.mkdir(parents=True, exist_ok=True)
        
        # إنشاء نسخة مؤقتة باسم نظيف وقصير لكي يعالجها demucs دون مشاكل مسارات
        temp_input = result_folder / f"{clean_stem}{input_path.suffix}"
        shutil.copy2(input_path, temp_input)

        vocals_audio = result_folder / "vocals.wav"
        music_audio = result_folder / "no_vocals.wav"
        
        detected_whisper_lang = "en"

        # ==========================================
        # STEP 1: DEMUCS AUDIO SEPARATION
        # ==========================================
        logger_callback("🎵 STEP 1/4: Running Demucs AI Audio Separation (Please wait)...")
        command = ["demucs", "--name", "htdemucs", "--out", str(self.output_dir), "--two-stems", "vocals", str(temp_input)]
        
        try:
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", creationflags=creationflags, env=env)
            if temp_input.exists():
                temp_input.unlink()
            if process.returncode != 0:
                logger_callback(f"❌ Demucs Error: {process.stderr}")
                return False
            logger_callback("✅ AI Audio Separation completed successfully!")
        except Exception as e:
            if temp_input.exists():
                temp_input.unlink()
            logger_callback(f"❌ Execution error during separation: {str(e)}")
            return False

        # ==========================================
        # STEP 2: FASTER-WHISPER TRANSCRIPTION
        # ==========================================
        original_srt_path = result_folder / f"{clean_stem}_original.srt"
        
        if "Subtitles" in mode:
            from faster_whisper import WhisperModel
            import torch
            
            w_device = "cuda" if hardware_type == "GPU" else "cpu"
            w_compute = "int8" if "INT8" in compute_type else "float16"
            if w_device == "cpu" and w_compute == "float16":
                w_compute = "int8" 
                
            logger_callback(f"📝 STEP 2/4: Initializing Faster-Whisper [{model_size.upper()} | {hardware_type} {gpu_index if hardware_type=='GPU' else ''} | {w_compute.upper()}]...")
            
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()

            try:
                dev_idx = [gpu_index] if w_device == "cuda" else 0
                whisper_model = WhisperModel(model_size, device=w_device, device_index=dev_idx, compute_type=w_compute)
                
                logger_callback("🎙️ Analyzing vocals and generating synchronized subtitles...")
                segments, info = whisper_model.transcribe(str(vocals_audio), beam_size=5)
                
                detected_whisper_lang = info.language
                logger_callback(f"🌍 Detected Language: {detected_whisper_lang.upper()} ({info.language_probability * 100:.2f}%)")

                subs = pysrt.SubRipFile()
                for i, segment in enumerate(segments):
                    start_time = pysrt.SubRipTime(seconds=segment.start)
                    end_time = pysrt.SubRipTime(seconds=segment.end)
                    sub = pysrt.SubRipItem(i+1, start=start_time, end=end_time, text=segment.text.strip())
                    subs.append(sub)
                
                subs.save(str(original_srt_path), encoding='utf-8')
                logger_callback("✅ Original Subtitles file generated (.srt) successfully.")

                del whisper_model
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                logger_callback(f"❌ Whisper Subtitling Error: {str(e)}")
                return False

        # ==========================================
        # STEP 3: NLLB META TRANSLATION
        # ==========================================
        translated_srt_path = result_folder / f"{clean_stem}_{target_lang}.srt"
        
        if "Translation" in mode and original_srt_path.exists():
            logger_callback(f"🌐 STEP 3/4: Loading Meta NLLB Neural Core for [{target_lang}]...")
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch
            import sentencepiece
            
            try:
                src_lang_code = self.whisper_to_nllb.get(detected_whisper_lang, "eng_Latn")
                target_code = self.lang_codes.get(target_lang, "eng_Latn")
                logger_callback(f"🧠 Dynamic Routing: Detected ({detected_whisper_lang}) -> Translating to ({target_lang})")

                nllb_model_name = "facebook/nllb-200-distilled-600M"
                tokenizer = AutoTokenizer.from_pretrained(nllb_model_name, src_lang=src_lang_code)
                model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name)
                
                nllb_device = f"cuda:{gpu_index}" if (hardware_type == "GPU" and HAS_CUDA) else "cpu"
                model = model.to(nllb_device)
                
                logger_callback("⚙️ Translating text blocks sequentially while preserving original timing...")
                subs = pysrt.open(str(original_srt_path), encoding='utf-8')
                
                target_lang_id = tokenizer.convert_tokens_to_ids(target_code)
                
                for sub in subs:
                    text = sub.text.strip().replace('\n', ' ')
                    if text: 
                        inputs = tokenizer(text, return_tensors="pt").to(nllb_device)
                        outputs = model.generate(
                            **inputs,
                            forced_bos_token_id=target_lang_id,
                            max_length=128
                        )
                        translated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                        sub.text = translated_text
                
                subs.save(str(translated_srt_path), encoding='utf-8')
                logger_callback(f"✅ Neural Translation to {target_lang} finished successfully.")

                del model, tokenizer
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                error_msg = f"❌ NLLB Translation Engine Error: {str(e)}"
                if hasattr(e, '__cause__') and e.__cause__:
                    error_msg += f"\n🔍 Hidden Real Cause: {str(e.__cause__)}"
                logger_callback(error_msg)
                return False

        # ==========================================
        # STEP 4: MULTI-STEM VIDEO MIXING
        # ==========================================
        video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.webm']
        if input_path.suffix.lower() in video_extensions:
            logger_callback("🎬 STEP 4/4: Executing final FFmpeg multiplexing (Injecting MKV Soft Subtitles)...")
            
            video_vocals_out = result_folder / "Video_VocalsOnly.mkv"
            video_music_out = result_folder / "Video_MusicOnly.mkv"
            
            def build_ffmpeg_cmd(input_vid, input_aud, output_vid):
                cmd = [ffmpeg_exe, "-y", "-i", str(input_vid), "-i", str(input_aud)]
                
                has_orig = original_srt_path.exists() and "Subtitles" in mode
                has_trans = translated_srt_path.exists() and "Translation" in mode
                
                if has_orig: cmd.extend(["-i", str(original_srt_path)])
                if has_trans: cmd.extend(["-i", str(translated_srt_path)])
                
                cmd.extend(["-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0"])
                
                sub_idx = 0
                if has_orig:
                    cmd.extend(["-map", "2:0", f"-metadata:s:s:{sub_idx}", f"language={detected_whisper_lang[:3]}"])
                    sub_idx += 1
                if has_trans:
                    idx = 3 if has_orig else 2
                    cmd.extend(["-map", f"{idx}:0", f"-metadata:s:s:{sub_idx}", f"language={target_lang[:3].lower()}"])
                    sub_idx += 1
                    
                if sub_idx > 0:
                    cmd.extend(["-c:s", "srt"])
                    
                cmd.extend(["-shortest", str(output_vid)])
                return cmd

            cmd_vocals = build_ffmpeg_cmd(input_path, vocals_audio, video_vocals_out)
            cmd_music = build_ffmpeg_cmd(input_path, music_audio, video_music_out)
            
            subprocess.run(cmd_vocals, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=creationflags, env=env)
            subprocess.run(cmd_music, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=creationflags, env=env)
            
            logger_callback("🌟 FFmpeg Studio Mixing complete! MKV Videos generated with embedded subtitles.")

        logger_callback(f"📁 Studio Process Finished! Target Folder:\n{result_folder}")
        if os.name == 'nt':
            os.startfile(result_folder)
        return True

# ==========================================
# 2. THE FRONTEND GUI (VOCAL REMOVER PRO)
# ==========================================
class VocalRemoverProApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Vocal Remover Pro - Ultimate AI Audio Studio")
        self.geometry("850x850")
        self.resizable(False, False)
        
        self.engine = AudioSeparatorEngine()
        self.selected_file = None
        
        self.hw_options = []
        self.vram_map = {} 
        self.scan_hardware()
        
        self.vram_requirements = {
            "base": {"INT8": 1.0, "FP16": 1.5},
            "medium": {"INT8": 2.5, "FP16": 3.5},
            "large-v3": {"INT8": 4.0, "FP16": 6.5} 
        }

        self.build_ui()
        self.validate_constraints()

    def scan_hardware(self):
        if HAS_CUDA:
            try:
                import torch
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    name = torch.cuda.get_device_name(i)
                    vram_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    label = f"[GPU {i}] {name} ({vram_gb:.1f} GB)"
                    self.hw_options.append(label)
                    self.vram_map[label] = vram_gb
            except Exception:
                pass
                
        threads = multiprocessing.cpu_count()
        cpu_label = f"[CPU] System Processor ({threads} Threads)"
        self.hw_options.append(cpu_label)
        self.vram_map[cpu_label] = 999.0 

    def build_ui(self):
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.pack(pady=(20, 10), fill="x")

        self.lbl_title = ctk.CTkLabel(self.header_frame, text="VOCAL REMOVER PRO", font=ctk.CTkFont(size=32, weight="bold"))
        self.lbl_title.pack()

        self.lbl_subtitle = ctk.CTkLabel(self.header_frame, text="Hardware-Aware Audio & Translation Studio", text_color="#A0A0A0", font=ctk.CTkFont(size=13))
        self.lbl_subtitle.pack()

        self.card_frame = ctk.CTkFrame(self, corner_radius=15, fg_color="#1E1E1E", border_width=1, border_color="#333333")
        self.card_frame.pack(pady=5, padx=30, fill="x")

        self.btn_browse = ctk.CTkButton(
            self.card_frame, text="📂 Browse Media File", width=250, height=45, 
            corner_radius=8, font=ctk.CTkFont(size=14, weight="bold"), 
            fg_color="#2A2A2A", hover_color="#3A3A3A", border_width=1, border_color="#555555",
            command=self.browse_file
        )
        self.btn_browse.pack(pady=(20, 5))

        self.lbl_file_path = ctk.CTkLabel(self.card_frame, text="No track loaded yet", text_color="#777777", font=ctk.CTkFont(size=12, slant="italic"))
        self.lbl_file_path.pack(pady=(0, 15))

        self.divider = ctk.CTkFrame(self.card_frame, height=2, fg_color="#333333")
        self.divider.pack(fill="x", padx=30, pady=(0, 15))

        self.settings_frame = ctk.CTkFrame(self.card_frame, fg_color="transparent")
        self.settings_frame.pack(pady=5, padx=40, fill="x")
        self.settings_frame.columnconfigure(1, weight=1)

        self.lbl_mode = ctk.CTkLabel(self.settings_frame, text="Processing Mode:", font=ctk.CTkFont(size=13, weight="bold"))
        self.lbl_mode.grid(row=0, column=0, padx=(0, 20), pady=5, sticky="e")

        self.combo_mode = ctk.CTkComboBox(
            self.settings_frame, 
            values=["1. Separation Only", "2. Separation + Subtitles", "3. Separation + Subtitles + Translation"],
            width=300, command=self.on_mode_change
        )
        self.combo_mode.grid(row=0, column=1, pady=5, sticky="w")
        self.combo_mode.set("3. Separation + Subtitles + Translation")

        self.lbl_lang = ctk.CTkLabel(self.settings_frame, text="Target Language:", font=ctk.CTkFont(size=13, weight="bold"))
        self.lbl_lang.grid(row=1, column=0, padx=(0, 20), pady=5, sticky="e")

        self.combo_lang = ctk.CTkComboBox(self.settings_frame, values=["Arabic", "English", "French", "Spanish", "German"], width=300)
        self.combo_lang.grid(row=1, column=1, pady=5, sticky="w")

        self.pro_switch_frame = ctk.CTkFrame(self.card_frame, fg_color="transparent")
        self.pro_switch_frame.pack(fill="x", padx=40, pady=(10, 0))
        
        self.switch_pro = ctk.CTkSwitch(self.pro_switch_frame, text="⚙️ Enable Pro Mode (Hardware Control)", font=ctk.CTkFont(weight="bold"), command=self.toggle_pro_mode)
        self.switch_pro.pack(side="left")

        self.pro_panel = ctk.CTkFrame(self.card_frame, corner_radius=10, fg_color="#151515", border_width=1, border_color="#444444")
        self.pro_panel.columnconfigure(0, weight=1)
        self.pro_panel.columnconfigure(1, weight=2)
        self.pro_panel.columnconfigure(2, weight=1)

        ctk.CTkLabel(self.pro_panel, text="AI Model Size", font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=0, pady=(10,0))
        self.combo_model = ctk.CTkComboBox(self.pro_panel, values=["Base (Fast)", "Medium (Balanced)", "Large-v3 (Accuracy)"], width=160, command=self.validate_constraints)
        self.combo_model.grid(row=1, column=0, padx=10, pady=(5,10))

        ctk.CTkLabel(self.pro_panel, text="Compute Device", font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=1, pady=(10,0))
        self.combo_hw = ctk.CTkComboBox(self.pro_panel, values=self.hw_options, width=280, command=self.validate_constraints)
        self.combo_hw.grid(row=1, column=1, padx=10, pady=(5,10))

        ctk.CTkLabel(self.pro_panel, text="VRAM Precision", font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=2, pady=(10,0))
        self.combo_precision = ctk.CTkComboBox(self.pro_panel, values=["INT8 (Low VRAM)", "FP16 (High Quality)"], width=160, command=self.validate_constraints)
        self.combo_precision.grid(row=1, column=2, padx=10, pady=(5,10))

        self.lbl_constraint = ctk.CTkLabel(self.card_frame, text="", text_color="#FF4500", font=ctk.CTkFont(weight="bold", size=13))

        self.action_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.action_frame.pack(pady=5, padx=40, fill="x")

        self.btn_start = ctk.CTkButton(
            self.action_frame, text="⚡ START ENGINE", height=50, 
            corner_radius=25, fg_color="#1F6AA5", hover_color="#144870", 
            font=ctk.CTkFont(size=16, weight="bold"), command=self.start_processing
        )
        self.btn_start.pack(fill="x")

        self.progress_bar = ctk.CTkProgressBar(self.action_frame, mode="indeterminate", height=6, corner_radius=3, fg_color="#333333", progress_color="#00BFFF")
        self.progress_bar.set(0)

        self.log_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#0A0A0A", border_width=1, border_color="#222222")
        self.log_frame.pack(pady=(10, 15), padx=30, fill="both", expand=True)

        self.txt_log = ctk.CTkTextbox(self.log_frame, font=ctk.CTkFont(family="Consolas", size=12), fg_color="transparent", text_color="#4AF626")
        self.txt_log.pack(padx=10, pady=10, fill="both", expand=True)
        
        hw_status = "CUDA Multi-GPU Matrix Loaded" if HAS_CUDA else "No GPU Detected - CPU Mode Active"
        self.txt_log.insert("0.0", f"[System] Hardware Profiler Initialized. [{hw_status}]\n")
        self.txt_log.configure(state="disabled")

    def toggle_pro_mode(self):
        if self.switch_pro.get() == 1:
            self.pro_panel.pack(fill="x", padx=30, pady=15)
            self.validate_constraints()
        else:
            self.pro_panel.pack_forget()
            self.lbl_constraint.pack_forget()
            self.btn_start.configure(state="normal", fg_color="#1F6AA5", text="⚡ START ENGINE")

    def validate_constraints(self, *args):
        if self.switch_pro.get() != 1: return
        
        model_key = self.combo_model.get().split()[0].lower()
        precision_key = "FP16" if "FP16" in self.combo_precision.get() else "INT8"
        hw_selected = self.combo_hw.get()
        
        if "[CPU]" in hw_selected:
            if precision_key == "FP16":
                self.lbl_constraint.configure(text="⚠️ CPU doesn't support FP16 well. System will force INT8.", text_color="#E1AD01")
                self.lbl_constraint.pack(pady=(0,10)) 
                self.btn_start.configure(state="normal", fg_color="#1F6AA5", text="⚡ START ENGINE")
            elif "large" in model_key:
                self.lbl_constraint.configure(text="⚠️ Warning: Large-v3 on CPU will be extremely slow (15m+ per song).", text_color="#E1AD01")
                self.lbl_constraint.pack(pady=(0,10))
                self.btn_start.configure(state="normal", fg_color="#1F6AA5", text="⚡ START ENGINE")
            else:
                self.lbl_constraint.pack_forget()
                self.btn_start.configure(state="normal", fg_color="#1F6AA5", text="⚡ START ENGINE")
            return

        if "[GPU" in hw_selected:
            required_vram = self.vram_requirements[model_key][precision_key]
            available_vram = self.vram_map[hw_selected]
            
            if required_vram > available_vram:
                msg = f"🚫 VRAM OVERLOAD: Requires ~{required_vram}GB, Your GPU has {available_vram:.1f}GB.\nPlease change Model to Medium, or Precision to INT8."
                self.lbl_constraint.configure(text=msg, text_color="#FF4500")
                self.lbl_constraint.pack(pady=(0,10))
                self.btn_start.configure(state="disabled", fg_color="#555555", text="🚫 HARDWARE LIMIT EXCEEDED")
            else:
                self.lbl_constraint.pack_forget()
                self.btn_start.configure(state="normal", fg_color="#1F6AA5", text="⚡ START ENGINE")

    def on_mode_change(self, choice):
        if "Translation" in choice:
            self.combo_lang.configure(state="normal", fg_color="#343638", text_color="white")
            self.lbl_lang.configure(text_color="white")
        else:
            self.combo_lang.configure(state="disabled", fg_color="#222222", text_color="#555555")
            self.lbl_lang.configure(text_color="#555555")

    def browse_file(self):
        filetypes = (("Supported Media", "*.mp3 *.wav *.flac *.m4a *.mp4 *.mkv *.webm *.avi *.mov"), ("All Files", "*.*"))
        filename = filedialog.askopenfilename(title="Select File", filetypes=filetypes)
        if filename:
            self.selected_file = filename
            self.lbl_file_path.configure(text=f"🎵 {Path(filename).name}", text_color="#00BFFF")
            self.log_message(f"[User] Target media loaded: {Path(filename).name}")

    def log_message(self, message):
        self.txt_log.configure(state="normal")
        self.txt_log.insert("end", message + "\n")
        self.txt_log.see("end")
        self.txt_log.configure(state="disabled")

    def start_processing(self):
        if not self.selected_file:
            messagebox.showwarning("Warning", "Please select a media file first.")
            return

        mode = self.combo_mode.get()
        target_lang = self.combo_lang.get()
        
        model_raw = self.combo_model.get()
        model_size = model_raw.split()[0].lower()
        compute_type = self.combo_precision.get()
        
        hw_selected = self.combo_hw.get()
        hardware_type = "CPU" if "[CPU]" in hw_selected else "GPU"
        
        gpu_index = 0
        if hardware_type == "GPU":
            start = hw_selected.find("[GPU ") + 5
            end = hw_selected.find("]", start)
            try:
                gpu_index = int(hw_selected[start:end])
            except ValueError:
                gpu_index = 0

        self.btn_start.configure(state="disabled", text="⚙️ NEURAL ENGINE ACTIVE...", fg_color="#444444")
        for widget in [self.btn_browse, self.combo_mode, self.combo_lang, self.switch_pro, self.combo_model, self.combo_hw, self.combo_precision]:
            widget.configure(state="disabled")
        
        self.progress_bar.pack(fill="x", padx=20, pady=(10,0))
        self.progress_bar.start()

        processing_thread = threading.Thread(target=self._run_engine_thread, args=(self.selected_file, mode, target_lang, model_size, hardware_type, gpu_index, compute_type))
        processing_thread.daemon = True
        processing_thread.start()

    def _run_engine_thread(self, file_path, mode, target_lang, model_size, hardware_type, gpu_index, compute_type):
        success = self.engine.process_pipeline(file_path, mode, target_lang, model_size, hardware_type, gpu_index, compute_type, self.log_message)
        self.after(0, self._finalize_ui, success)

    def _finalize_ui(self, success):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()

        if success:
            self.btn_start.configure(state="normal", text="✨ WORK COMPLETE! LOAD NEW FILE", fg_color="#1F6AA5")
            self.selected_file = None
            self.lbl_file_path.configure(text="No track loaded yet", text_color="#777777")
        else:
            self.btn_start.configure(state="normal", text="🔄 RETRY FAILED PIPELINE", fg_color="#CC0000")
        
        for widget in [self.btn_browse, self.combo_mode, self.switch_pro, self.combo_model, self.combo_hw, self.combo_precision]:
            widget.configure(state="normal")
        self.on_mode_change(self.combo_mode.get())
        if self.switch_pro.get() == 1: self.validate_constraints()

if __name__ == "__main__":
    app = VocalRemoverProApp()
    app.mainloop()