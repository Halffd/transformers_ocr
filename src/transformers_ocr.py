#!/usr/bin/python3
# Copyright: Ren Tatsumoto <tatsu at autistici.org> and contributors
# License: GNU GPL, version 3 or later; http://www.gnu.org/licenses/gpl.html

import argparse
import dataclasses
import datetime
import enum
import json
import os
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from argparse import RawTextHelpFormatter
from typing import AnyStr, IO, Iterable, Optional
import fcntl
import socket
import struct
import pickle
from pathlib import Path

# Global arguments variable
args = None

MANGA_OCR_PREFIX = os.path.join(os.environ["HOME"], ".local", "share", "manga_ocr")
MANGA_OCR_PYENV_PATH = os.path.join(MANGA_OCR_PREFIX, "pyenv")
MANGA_OCR_PYENV_PIP_PATH = os.path.join(MANGA_OCR_PYENV_PATH, "bin", "pip")
HUGGING_FACE_CACHE_PATH = os.path.join(os.environ["HOME"], '.cache', 'huggingface')
CONFIG_PATH = os.path.join(
    os.environ.get("XDG_CONFIG_HOME", os.path.join(os.environ["HOME"], ".config")),
    "transformers_ocr",
    "config",
)
SOCKET_PATH = "/tmp/manga_ocr.sock"
PID_FILE = "/tmp/manga_ocr.pid"
PROGRAM = "transformers_ocr"
JOIN = "、"
IS_XORG = "WAYLAND_DISPLAY" not in os.environ
IS_GNOME = os.environ.get("XDG_CURRENT_DESKTOP") == "GNOME"
IS_KDE = os.environ.get("XDG_CURRENT_DESKTOP") == "KDE"
IS_XFCE = os.environ.get("XDG_CURRENT_DESKTOP") == "XFCE"
CLIP_COPY_ARGS = (
    ("xclip", "-selection", "clipboard",)
    if IS_XORG
    else ("wl-copy",)
)
CLIP_TEXT_PLACEHOLDER = '%TEXT%'

# Add path to local manga-ocr
MANGA_OCR_SRC_PATH = os.path.join(os.path.dirname(__file__), "manga-ocr")
if os.path.exists(MANGA_OCR_SRC_PATH):
    sys.path.insert(0, MANGA_OCR_SRC_PATH)

class Platform(enum.Enum):
    GNOME = enum.auto()
    KDE = enum.auto()
    XFCE = enum.auto()
    Xorg = enum.auto()
    Wayland = enum.auto()

    @classmethod
    def current(cls):
        if IS_GNOME:
            return cls.GNOME
        elif IS_KDE:
            return cls.KDE
        elif IS_XFCE:
            return cls.XFCE
        elif IS_XORG:
            return cls.Xorg
        else:
            return cls.Wayland


CURRENT_PLATFORM = Platform.current()


class MissingProgram(RuntimeError):
    pass


class StopRequested(Exception):
    pass


class ScreenshotCancelled(RuntimeError):
    pass


@dataclasses.dataclass
class OcrCommand:
    action: str
    file_path: str | None

    def as_json(self):
        return json.dumps(dataclasses.asdict(self))


def is_pacman_installed(program: str) -> bool:
    return subprocess.call(("pacman", "-Qq", program,), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, ) == 0


def is_installed(program: str) -> bool:
    return shutil.which(program) or is_pacman_installed(program)


def raise_if_missing(*programs):
    for prog in programs:
        if not is_installed(prog):
            raise MissingProgram(f"{prog} must be installed for {PROGRAM} to work.")


def gnome_screenshot_select(screenshot_path: str):
    raise_if_missing("gnome-screenshot")
    return subprocess.run(
        ("gnome-screenshot", "-a", "-f", screenshot_path,),
        check=True,
    )


def spectactle_select(screenshot_path: str):
    raise_if_missing("spectacle")
    return subprocess.run(
        ("spectacle", "-n", "-b", "-r", "-o", screenshot_path,),
        check=True,
        stderr=subprocess.DEVNULL
    )

def xfce_screenshooter_select(screenshot_path: str):
    raise_if_missing("xfce4-screenshooter")
    return subprocess.run(
        ("xfce4-screenshooter", "-r", "-s", screenshot_path,),
        check=True,
        stderr=subprocess.DEVNULL
    )

def maim_select(screenshot_path: str):
    raise_if_missing("maim")
    return subprocess.run(
        ("maim", "--select", "--hidecursor", "--format=png", "--quality", "1", screenshot_path,),
        check=True,
        stderr=sys.stdout,
    )


def grim_select(screenshot_path: str):
    raise_if_missing("grim", "slurp")
    return subprocess.run(
        ("grim", "-g", subprocess.check_output(["slurp"]).decode().strip(), screenshot_path,),
        check=True,
    )


def gnome_screenshot_full(screenshot_path: str):
    raise_if_missing("gnome-screenshot")
    return subprocess.run(
        ("gnome-screenshot", "-f", screenshot_path,),
        check=True,
    )


def spectacle_full(screenshot_path: str):
    raise_if_missing("spectacle")
    return subprocess.run(
        ("spectacle", "-n", "-b", "-f", "-o", screenshot_path,),
        check=True,
        stderr=subprocess.DEVNULL
    )


def xfce_screenshooter_full(screenshot_path: str):
    raise_if_missing("xfce4-screenshooter")
    return subprocess.run(
        ("xfce4-screenshooter", "-f", "-s", screenshot_path,),
        check=True,
        stderr=subprocess.DEVNULL
    )


def maim_full(screenshot_path: str):
    raise_if_missing("maim")
    return subprocess.run(
        ("maim", "--format=png", "--quality", "1", screenshot_path,),
        check=True,
        stderr=sys.stdout,
    )


def grim_full(screenshot_path: str):
    raise_if_missing("grim")
    return subprocess.run(
        ("grim", screenshot_path,),
        check=True,
    )


def gnome_screenshot_active(screenshot_path: str):
    raise_if_missing("gnome-screenshot")
    return subprocess.run(
        ("gnome-screenshot", "-w", "-f", screenshot_path,),
        check=True,
    )


def spectacle_active(screenshot_path: str):
    raise_if_missing("spectacle")
    return subprocess.run(
        ("spectacle", "-n", "-b", "-a", "-o", screenshot_path,),
        check=True,
        stderr=subprocess.DEVNULL
    )


def xfce_screenshooter_active(screenshot_path: str):
    raise_if_missing("xfce4-screenshooter")
    return subprocess.run(
        ("xfce4-screenshooter", "-w", "-s", screenshot_path,),
        check=True,
        stderr=subprocess.DEVNULL
    )


def maim_active(screenshot_path: str):
    raise_if_missing("maim", "xdotool")
    try:
        window_id = subprocess.check_output(["xdotool", "getactivewindow"]).decode().strip()
        return subprocess.run(
            ("maim", "--window", window_id, "--format=png", "--quality", "1", screenshot_path,),
            check=True,
            stderr=sys.stdout,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to full screen if can't get active window
        return maim_full(screenshot_path)


def grim_active(screenshot_path: str):
    try:
        raise_if_missing("grim", "swaymsg", "jq")
        # Get active output
        output_cmd = 'swaymsg -t get_outputs | jq -r \'.[] | select(.focused) | .name\''
        active_output = subprocess.check_output(output_cmd, shell=True).decode().strip()
        if active_output:
            return subprocess.run(
                ("grim", "-o", active_output, screenshot_path,),
                check=True,
            )
        else:
            return grim_full(screenshot_path)
    except (subprocess.CalledProcessError, FileNotFoundError, MissingProgram):
        # Fallback to full screen
        return grim_full(screenshot_path)


def take_screenshot(screenshot_path, full_screen=False, active_monitor=False):
    if active_monitor:
        match CURRENT_PLATFORM:
            case Platform.GNOME:
                gnome_screenshot_active(screenshot_path)
            case Platform.KDE:
                spectacle_active(screenshot_path)
            case Platform.XFCE:
                xfce_screenshooter_active(screenshot_path)
            case Platform.Xorg:
                maim_active(screenshot_path)
            case Platform.Wayland:
                grim_active(screenshot_path)
    elif full_screen:
        match CURRENT_PLATFORM:
            case Platform.GNOME:
                gnome_screenshot_full(screenshot_path)
            case Platform.KDE:
                spectacle_full(screenshot_path)
            case Platform.XFCE:
                xfce_screenshooter_full(screenshot_path)
            case Platform.Xorg:
                maim_full(screenshot_path)
            case Platform.Wayland:
                grim_full(screenshot_path)
    else:
        match CURRENT_PLATFORM:
            case Platform.GNOME:
                gnome_screenshot_select(screenshot_path)
            case Platform.KDE:
                spectactle_select(screenshot_path)
            case Platform.XFCE:
                xfce_screenshooter_select(screenshot_path)
            case Platform.Xorg:
                maim_select(screenshot_path)
            case Platform.Wayland:
                grim_select(screenshot_path)


def prepare_pipe():
    # Remove this function as we're using sockets now
    pass


def write_command_to_pipe(command: str, file_path: str):
    client = OcrClient(debug=args.debug)
    try:
        client.send_command(OcrCommand(action=command, file_path=file_path))
    except Exception as e:
        raise IOError(f"Failed to send command: {e}")


def run_ocr(command: str, image_path: Optional[str] = None) -> None:
    try:
        ensure_listening()
        
        if image_path is not None:
            write_command_to_pipe(command, image_path)
            return

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as screenshot_file:
            try:
                take_screenshot(screenshot_file.name)
            except subprocess.CalledProcessError as ex:
                raise ScreenshotCancelled() from ex
                
            # Add small delay to ensure file is written
            time.sleep(0.1)
            
            try:
                write_command_to_pipe(command, screenshot_file.name)
            except IOError as e:
                notify_send(f"Failed to communicate with OCR service: {e}")
                # Cleanup
                os.unlink(screenshot_file.name)
                raise
    except KeyboardInterrupt:
        notify_send("OCR cancelled by user")
        sys.exit(1)


def is_running(pid: int) -> bool:
    return pid > 0 and os.path.exists(f"/proc/{pid}")


def get_pid() -> int | None:
    try:
        with open(PID_FILE) as pid_file:
            pid = int(pid_file.read())
    except (ValueError, FileNotFoundError):
        return None
    else:
        return pid if is_running(pid) else None


def debug_log(msg: str, debug_enabled: bool = False):
    if debug_enabled:
        print(f"[DEBUG] {msg}")


def ensure_listening():
    if os.path.exists(MANGA_OCR_PREFIX):
        if get_pid() is None:
            debug_log("Starting manga_ocr listener...", args.debug)
            
            python_path = os.path.join(MANGA_OCR_PYENV_PATH, "bin", "python3")
            if not os.path.exists(python_path):
                print("Virtual environment not found. Please run 'download' first.")
                sys.exit(1)
            
            cmd = [python_path, __file__]
            if args.debug:
                cmd.append("-d")
            cmd.extend(["start", "--foreground"])
            
            debug_log(f"Running command: {' '.join(cmd)}", args.debug)
            
            # Start server in background
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE if not args.debug else None,
                stderr=subprocess.PIPE if not args.debug else None,
                start_new_session=True
            )
            
            # Write PID file
            with open(PID_FILE, "w") as pid_file:
                pid_file.write(str(p.pid))
            
            # Wait for socket to be available
            start_time = time.time()
            while time.time() - start_time < 600:
                if os.path.exists(SOCKET_PATH):
                    try:
                        # Test connection
                        client = OcrClient(debug=args.debug)
                        client.send_command(OcrCommand("ping", None))
                        debug_log("Server is ready", args.debug)
                        return
                    except IOError as e:
                        debug_log(f"Server not ready yet: {e}", args.debug)
                time.sleep(1)
                
            # If we get here, server failed to start
            p.kill()
            raise TimeoutError("Server failed to start")
        else:
            if os.path.exists(SOCKET_PATH):
                print("Already running.")
            else:
                print("Process exists but socket is missing. Restarting...")
                stop_listening()
                time.sleep(1)
                ensure_listening()
    else:
        print("manga-ocr is not downloaded.")
        sys.exit(1)


def kill_after(pid: int, timeout_s: float, step_s: float = 0.1):
    for _step in range(int(timeout_s // step_s)):
        if get_pid() is None:
            break
        time.sleep(step_s)
        print(".", end="", flush=True)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        print(" Stopped.")
    else:
        print(" Killed.")


def stop_listening():
    if (pid := get_pid()) is not None:
        try:
            # Try to send stop command
            write_command_to_pipe("stop", None)
            kill_after(pid, timeout_s=3)
        except (IOError, TimeoutError):
            # If pipe write fails, force kill
            try:
                os.kill(pid, signal.SIGKILL)
                print("Force killed unresponsive process.")
            except ProcessLookupError:
                pass
        
        # Clean up files
        for file in (PID_FILE, SOCKET_PATH):
            try:
                os.remove(file)
            except FileNotFoundError:
                pass
    else:
        print("Already stopped.")


def is_fifo(path: AnyStr) -> bool:
    try:
        return stat.S_ISFIFO(os.stat(path).st_mode)
    except FileNotFoundError:
        return False


def notify_send(msg: str):
    print(msg)
    try:
        # First attempt: standard notify-send
        subprocess.Popen(("notify-send", "manga-ocr", msg,), shell=False)
    except (FileNotFoundError, OSError):
        try:
            # Fallback: Try with gdbus (works on many systems)
            subprocess.Popen([
                "gdbus", "call", "--session", "--dest", "org.freedesktop.Notifications",
                "--object-path", "/org/freedesktop/Notifications",
                "--method", "org.freedesktop.Notifications.Notify",
                "manga-ocr", "0", "", "manga-ocr", msg, "[]", "{}", "5000"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (FileNotFoundError, OSError):
            # If everything fails, just print to console (already done above)
            pass


def is_valid_key_val_pair(line: str) -> bool:
    return "=" in line and not line.startswith("#")


def get_config() -> dict[str, str]:
    config = {}
    if os.path.isfile(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf8") as f:
            for line in filter(is_valid_key_val_pair, f.read().splitlines()):
                key, value = line.split("=", maxsplit=1)
                config[key] = value
    return config


class TrOcrConfig:
    def __init__(self):
        self._config = get_config()
        self.force_cpu = self._should_force_cpu()
        self.clip_args = self._key_to_cmd_args("clip_command")
        self.screenshot_dir = self._get_screenshot_dir()

    def _should_force_cpu(self) -> bool:
        return bool(self._config.get('force_cpu', 'no') in ('true', 'yes',))

    def _key_to_cmd_args(self, key: str) -> list[str] | None:
        try:
            return shlex.split(self._config[key].strip())
        except (KeyError, AttributeError):
            return None

    def _get_screenshot_dir(self):
        if (screenshot_dir := self._config.get('screenshot_dir')) and os.path.isdir(screenshot_dir):
            return screenshot_dir


def iter_commands(stream: IO) -> Iterable[OcrCommand]:
    yield from (OcrCommand(**json.loads(line)) for line in stream)


class MangaOcrWrapper:
    def __init__(self, debug=False):
        self._config = TrOcrConfig()
        self.debug = debug
        self._debug_log("Initializing MangaOCR model...")
        
        start_time = time.time()
        try:
            if MANGA_OCR_SRC_PATH in sys.path:
                self._debug_log(f"Attempting to load local manga-ocr from {MANGA_OCR_SRC_PATH}")
                from manga_ocr import MangaOcr
            else:
                self._debug_log("Loading system manga-ocr")
                from manga_ocr import MangaOcr
                
            self._mocr = MangaOcr(force_cpu=self._config.force_cpu)
            init_time = time.time() - start_time
            self._debug_log(f"Model initialized in {init_time:.2f}s")
        except ImportError as e:
            self._debug_log(f"Failed to import manga-ocr: {e}")
            raise RuntimeError("Failed to initialize manga-ocr. Is it installed?")
        
        self._on_hold = []
        self._last_ocr_time = 0
        self._min_interval = 0.1
        
        # Cache the model's device
        self._device = next(self._mocr.model.parameters()).device
        self._debug_log(f"Model running on device: {self._device}")

    def _debug_log(self, msg: str):
        if self.debug:
            print(f"[DEBUG] {msg}")

    def _ocr(self, file_path: str) -> str:
        try:
            # Add rate limiting
            current_time = time.time()
            if current_time - self._last_ocr_time < self._min_interval:
                wait_time = self._min_interval - (current_time - self._last_ocr_time)
                self._debug_log(f"Rate limiting: waiting {wait_time:.2f}s")
                time.sleep(wait_time)
            
            # Verify file exists and has content
            self._debug_log(f"Processing file: {file_path}")
            if not os.path.exists(file_path):
                msg = "Error: Screenshot file not found"
                self._debug_log(msg)
                notify_send(msg)
                return ""
                
            file_size = os.path.getsize(file_path)
            self._debug_log(f"File size: {file_size} bytes")
            if file_size == 0:
                msg = "Error: Empty screenshot file"
                self._debug_log(msg)
                notify_send(msg)
                return ""

            # Perform OCR with detailed timing
            self._debug_log("Starting OCR...")
            start_time = time.time()
            
            # Break down OCR steps for debugging
            try:
                self._debug_log("Loading image...")
                text = self._mocr(file_path)
            except Exception as e:
                self._debug_log(f"OCR internal error: {str(e)}\n{traceback.format_exc()}")
                raise
            
            ocr_time = time.time() - start_time
            self._last_ocr_time = time.time()
            self._debug_log(f"OCR completed in {ocr_time:.2f}s")
            
            # Validate output
            if not text or len(text.strip()) == 0:
                msg = "Warning: No text detected"
                self._debug_log(msg)
                notify_send(msg)
                return ""
            
            self._debug_log(f"Raw OCR text: {text}")
            processed_text = (
                text.strip()
                .replace('...', '…')
                .replace('。。。', '…')
                .replace('．．．', '…')
            )
            self._debug_log(f"Processed text: {processed_text}")
            return processed_text

        except Exception as e:
            msg = f"OCR Error: {str(e)}"
            self._debug_log(f"{msg}\n{traceback.format_exc()}")
            notify_send(msg)
            return ""

    def _process_command(self, command: OcrCommand):
        if command.action == "ping":
            return
            
        if command.action == "stop":
            raise StopRequested()
            
        if not command.file_path or not os.path.exists(command.file_path):
            self._debug_log(f"File not found: {command.file_path}")
            return
        
        # Special handling for "select" action
        if command.action == "select":
            self._display_image_for_selection(command.file_path)
            return
            
        try:
            text = self._ocr(command.file_path)
            if command.action == "hold":
                self._on_hold.append(text)
            else:
                self._copy_to_clipboard(text)
                notify_send(text)
        finally:
            if command.file_path and os.path.exists(command.file_path) and command.action != "select":
                os.unlink(command.file_path)
                
    def _display_image_for_selection(self, image_path):
        self._debug_log(f"Displaying image for selection: {image_path}")
        
        # Check if file exists and has content
        if not os.path.exists(image_path):
            self._debug_log(f"Error: Screenshot file not found at {image_path}")
            notify_send("Error: Screenshot file not found")
            return
        
        file_size = os.path.getsize(image_path)
        if file_size == 0:
            self._debug_log(f"Error: Empty screenshot file at {image_path}")
            notify_send("Error: Empty screenshot file")
            return
            
        self._debug_log(f"Screenshot file is valid: {file_size} bytes")
        
        # Try to add a helpful message to the image
        temp_image_path = None
        try:
            if shutil.which("convert"):  # ImageMagick's convert
                self._debug_log("Adding text overlay with ImageMagick")
                temp_image_path = f"{image_path}.overlay.png"
                
                convert_cmd = [
                    "convert", image_path,
                    "-fill", "white", "-undercolor", "#00000080",
                    "-gravity", "North", "-pointsize", "24",
                    "-annotate", "+0+30", "SCREEN FROZEN - PRESS ESC OR Q TO EXIT",
                    temp_image_path
                ]
                self._debug_log(f"Running convert command: {' '.join(convert_cmd)}")
                
                # Create a copy with text overlay
                result = subprocess.run(
                    convert_cmd,
                    check=True, 
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE
                )
                
                # Use the modified image if successful
                if os.path.exists(temp_image_path) and os.path.getsize(temp_image_path) > 0:
                    self._debug_log(f"Created overlay image at {temp_image_path}")
                    image_path = temp_image_path
                else:
                    self._debug_log(f"Overlay image not created or empty, using original")
            else:
                self._debug_log("ImageMagick convert not found, skipping text overlay")
        except Exception as e:
            self._debug_log(f"Failed to add text overlay: {e}\n{traceback.format_exc()}")
            # Continue with original image if text overlay fails
        
        try:
            # Display image fullscreen to create a freeze effect
            viewer_cmd = None
            
            # Try each viewer in order of preference
            if shutil.which("feh"):
                self._debug_log("Using feh as image viewer")
                viewer_cmd = ["feh", "--fullscreen", "--borderless", "--hide-pointer", "--auto-zoom", image_path]
            elif shutil.which("mpv"):
                self._debug_log("Using mpv as image viewer")
                viewer_cmd = ["mpv", "--fullscreen", "--keep-open=yes", "--image-display-duration=inf", image_path]
            elif shutil.which("eog"):
                self._debug_log("Using eog (Eye of GNOME) as image viewer")
                viewer_cmd = ["eog", "--fullscreen", image_path]
            elif shutil.which("xdg-open"):
                self._debug_log("Using xdg-open as image viewer")
                viewer_cmd = ["xdg-open", image_path]
            
            if not viewer_cmd:
                self._debug_log("No suitable image viewer found")
                notify_send("No suitable image viewer found. Install feh, mpv, or eog for best results.")
                return
            
            self._debug_log(f"Launching viewer with command: {' '.join(viewer_cmd)}")
            
            # Use Popen with start_new_session=True to detach from parent process
            process = subprocess.Popen(
                viewer_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
            
            # Check if process started successfully
            time.sleep(0.5)
            if process.poll() is not None:
                stderr = process.stderr.read().decode('utf-8', errors='ignore')
                stdout = process.stdout.read().decode('utf-8', errors='ignore')
                self._debug_log(f"Viewer failed to start with return code {process.returncode}")
                self._debug_log(f"STDOUT: {stdout}")
                self._debug_log(f"STDERR: {stderr}")
                notify_send(f"Failed to display image: Viewer exited immediately with code {process.returncode}")
                return
                
            self._debug_log(f"Viewer process started with PID {process.pid}")
            
            # Notify based on which viewer was used
            if "feh" in viewer_cmd[0]:
                notify_send("Screen frozen. Press q or Escape to unfreeze.")
            elif "mpv" in viewer_cmd[0]:
                notify_send("Screen frozen. Press q or Escape to unfreeze.")
            elif "eog" in viewer_cmd[0]:
                notify_send("Screen frozen. Press F11 or Escape to unfreeze.")
            else:
                notify_send("Screen frozen. Close the viewer to unfreeze.")
            
        except Exception as e:
            self._debug_log(f"Error displaying image: {e}\n{traceback.format_exc()}")
            notify_send(f"Error displaying image: {e}")
        finally:
            # Clean up temporary file if created
            if temp_image_path and os.path.exists(temp_image_path) and temp_image_path != image_path:
                try:
                    os.unlink(temp_image_path)
                except OSError:
                    pass

    def _copy_to_clipboard(self, text: str):
        cmd_args = list(self._config.clip_args or CLIP_COPY_ARGS)
        try:
            placeholder_position = cmd_args.index(CLIP_TEXT_PLACEHOLDER)
            pass_text_to_stdin = False
            cmd_args[placeholder_position] = text
        except ValueError:
            pass_text_to_stdin = True

        try:
            raise_if_missing(cmd_args[0])
            p = subprocess.Popen(
                cmd_args,
                stdin=(subprocess.PIPE if pass_text_to_stdin else None),
                shell=False,
                start_new_session=True,
            )
            if pass_text_to_stdin:
                p.communicate(input=text.encode())
        except MissingProgram as ex:
            notify_send(str(ex))
        else:
            notify_send(f"Copied {text}")

    def _maybe_save_result(self, file_path, text):
        if self._config.screenshot_dir:
            name_pattern = datetime.datetime.now().strftime("trocr_%Y%m%d_%H%M%S")
            text_file_path = os.path.join(self._config.screenshot_dir, f'{name_pattern}.gt.txt')
            png_file_path = os.path.join(self._config.screenshot_dir, f'{name_pattern}.png')
            shutil.copy(file_path, png_file_path)
            with open(text_file_path, 'w', encoding='utf8') as of:
                of.write(text)

    def loop(self):
        try:
            while True:
                with open(SOCKET_PATH) as fifo:
                    for command in iter_commands(fifo):
                        self._process_command(command)
        except StopRequested:
            return notify_send("Stopped listening.")


class OcrServer:
    def __init__(self, debug=False):
        self.debug = debug
        self._wrapper = None  # Initialize later
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
    def _debug_log(self, msg):
        if self.debug:
            print(f"[DEBUG] {msg}")
            
    def start(self):
        # Initialize model first
        if not self._wrapper:
            self._wrapper = MangaOcrWrapper(debug=self.debug)
        
        # Remove existing socket file
        try:
            os.unlink(SOCKET_PATH)
        except OSError:
            if os.path.exists(SOCKET_PATH):
                raise

        self.sock.bind(SOCKET_PATH)
        self.sock.listen(1)
        os.chmod(SOCKET_PATH, 0o666)  # Allow other users to connect
        self._debug_log("Server listening on socket")
        
        try:
            while True:
                try:
                    conn, addr = self.sock.accept()
                    self._debug_log(f"Accepted connection")
                    self._handle_connection(conn)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self._debug_log(f"Error in connection: {e}")
                    continue
        finally:
            self._debug_log("Server shutting down")
            self.sock.close()
            try:
                os.unlink(SOCKET_PATH)
            except OSError:
                pass

    def _handle_connection(self, conn):
        try:
            # First read message length (4 bytes)
            msg_len_data = conn.recv(4)
            if not msg_len_data:
                return
                
            msg_len = struct.unpack('>I', msg_len_data)[0]
            self._debug_log(f"Receiving message of length {msg_len}")
            
            # Then read the message
            data = b''
            while len(data) < msg_len:
                chunk = conn.recv(min(msg_len - len(data), 4096))
                if not chunk:
                    break
                data += chunk
            
            if len(data) < msg_len:
                self._debug_log("Incomplete message received")
                return
                
            command = pickle.loads(data)
            self._debug_log(f"Received command: {command}")
            
            if command.action == "stop":
                self._debug_log("Received stop command")
                conn.close()
                self.sock.close()
                os.unlink(SOCKET_PATH)
                sys.exit(0)
                
            if command.action == "select":
                self._debug_log(f"Received select command with file: {command.file_path}")
                if os.path.exists(command.file_path):
                    self._debug_log(f"File exists and is {os.path.getsize(command.file_path)} bytes")
                else:
                    self._debug_log(f"File does not exist: {command.file_path}")
            
            try:
                self._debug_log(f"Processing command: {command.action}")
                self._wrapper._process_command(command)
                self._debug_log(f"Command {command.action} processed successfully")
            except Exception as e:
                self._debug_log(f"Error processing command {command.action}: {e}\n{traceback.format_exc()}")
            
        except Exception as e:
            self._debug_log(f"Error handling connection: {e}\n{traceback.format_exc()}")
        finally:
            conn.close()


class OcrClient:
    def __init__(self, debug=False):
        self.debug = debug
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        
    def _debug_log(self, msg):
        if self.debug:
            print(f"[DEBUG] {msg}")
            
    def send_command(self, command: OcrCommand, timeout=5):
        try:
            self.sock.settimeout(timeout)
            self.sock.connect(SOCKET_PATH)
            
            data = pickle.dumps(command)
            msg_len = struct.pack('>I', len(data))
            
            # Send length prefix and data
            self.sock.sendall(msg_len)
            self.sock.sendall(data)
            
            self._debug_log(f"Sent command: {command}")
        except socket.timeout:
            raise IOError("Connection timed out")
        except ConnectionRefusedError:
            raise IOError("Server not running")
        finally:
            self.sock.close()


def start_listening(args):
    if args.foreground:
        server = OcrServer(debug=args.debug)
        try:
            server.start()
        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            try:
                os.unlink(SOCKET_PATH)
            except OSError:
                pass
    else:
        ensure_listening()


def restart_listener():
    stop_listening()
    ensure_listening()


def status_str():
    return "Running" if get_pid() else "Stopped"


def print_status():
    print(f"{status_str()}, {CURRENT_PLATFORM.name}.")


def download_manga_ocr():
    print("Downloading manga-ocr and dependencies...")
    os.makedirs(MANGA_OCR_PREFIX, exist_ok=True)
    
    # Create virtual environment
    subprocess.run(
        ("python3", "-m", "venv", "--symlinks", MANGA_OCR_PYENV_PATH,),
        check=True,
    )
    
    # Upgrade pip
    subprocess.run(
        (MANGA_OCR_PYENV_PIP_PATH, "install", "--upgrade", "pip",),
        check=True,
    )
    
    # Install PyTorch first
    print("Installing PyTorch...")
    subprocess.run(
        (MANGA_OCR_PYENV_PIP_PATH, "install", "torch", "torchvision",),
        check=True,
    )
    
    # Install transformers
    print("Installing transformers...")
    subprocess.run(
        (MANGA_OCR_PYENV_PIP_PATH, "install", "transformers",),
        check=True,
    )
    
    # Install manga-ocr
    print("Installing manga-ocr...")
    subprocess.run(
        (MANGA_OCR_PYENV_PIP_PATH, "install", "--upgrade", "manga-ocr",),
        check=True,
    )
    
    print("Downloaded manga-ocr and all dependencies.")


def prog_name():
    return os.path.basename(sys.argv[0])


def purge_manga_ocr_data():
    shutil.rmtree(MANGA_OCR_PREFIX, ignore_errors=True)
    shutil.rmtree(HUGGING_FACE_CACHE_PATH, ignore_errors=True)
    print("Purged all downloaded manga-ocr data.")


def try_alternative_screenshot(screenshot_path: str, debug=False):
    """
    Try alternative methods to take a screenshot if the primary methods are failing.
    Returns True if successful, False otherwise.
    """
    try:
        # For X11/Xorg systems, try using import from ImageMagick which is often more reliable
        if IS_XORG and shutil.which("import"):
            debug_log("Trying ImageMagick import for screenshot", debug)
            subprocess.run(
                ["import", "-window", "root", screenshot_path],
                check=True,
                stderr=subprocess.DEVNULL
            )
            return True
            
        # For Wayland, try a more direct approach with grim if available
        if not IS_XORG and shutil.which("grim"):
            debug_log("Trying direct grim for screenshot", debug)
            subprocess.run(
                ["grim", screenshot_path],
                check=True, 
                stderr=subprocess.DEVNULL
            )
            return True
            
        # If scrot is available, try that as well (works on most X11 systems)
        if shutil.which("scrot"):
            debug_log("Trying scrot for screenshot", debug)
            subprocess.run(
                ["scrot", "-o", screenshot_path],
                check=True,
                stderr=subprocess.DEVNULL
            )
            return True
            
        return False
    except subprocess.CalledProcessError:
        return False


def run_screenshot_copy(full_screen=False):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as screenshot_file:
        try:
            take_screenshot(screenshot_file.name, full_screen=full_screen)
        except subprocess.CalledProcessError as ex:
            raise ScreenshotCancelled() from ex
            
        # Add small delay to ensure file is written
        time.sleep(0.1)
        
        # Copy to clipboard
        cmd_args = list(TrOcrConfig()._config.get('clip_command_image', CLIP_COPY_ARGS) or CLIP_COPY_ARGS)
        try:
            if IS_XORG:
                raise_if_missing(cmd_args[0])
                subprocess.run(
                    ["xclip", "-selection", "clipboard", "-t", "image/png", "-i", screenshot_file.name],
                    check=True
                )
            else:
                raise_if_missing("wl-copy")
                subprocess.run(
                    ["wl-copy", "-t", "image/png", screenshot_file.name],
                    check=True
                )
            notify_send("Screenshot copied to clipboard")
        except (MissingProgram, subprocess.CalledProcessError) as ex:
            notify_send(f"Failed to copy screenshot: {ex}")
        finally:
            # Clean up the temp file
            os.unlink(screenshot_file.name)
    
def run_select_freeze(image_path=None):
    try:
        debug_log("Starting select_freeze operation", args.debug)
        ensure_listening()
        
        if image_path is not None:
            debug_log(f"Using provided image: {image_path}", args.debug)
            # Use the provided image directly
            write_command_to_pipe("select", image_path)
            return

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as screenshot_file:
            debug_log(f"Creating screenshot at: {screenshot_file.name}", args.debug)
            
            screenshot_success = False
            try:
                # First try the standard method
                debug_log("Taking screenshot of active monitor", args.debug)
                take_screenshot(screenshot_file.name, active_monitor=True)
                screenshot_success = True
                debug_log("Standard screenshot successful", args.debug)
            except subprocess.CalledProcessError:
                debug_log("Standard screenshot method failed, trying alternatives", args.debug)
                pass
            
            # If the primary method fails, try alternative methods
            if not screenshot_success or not os.path.exists(screenshot_file.name) or os.path.getsize(screenshot_file.name) == 0:
                debug_log("Trying alternative screenshot methods", args.debug)
                screenshot_success = try_alternative_screenshot(screenshot_file.name, args.debug)
                
            # Verify the screenshot was created successfully
            if not screenshot_success or not os.path.exists(screenshot_file.name) or os.path.getsize(screenshot_file.name) == 0:
                notify_send("Failed to create screenshot. Make sure you have screenshot tools installed.")
                debug_log("All screenshot methods failed", args.debug)
                return
                
            debug_log(f"Screenshot created successfully: {os.path.getsize(screenshot_file.name)} bytes", args.debug)
                
            # Always save a debug copy when in debug mode
            if args.debug:
                debug_copy = "/tmp/manga_ocr_debug_screenshot.png"
                debug_log(f"Saving debug copy to {debug_copy}", args.debug)
                try:
                    shutil.copy2(screenshot_file.name, debug_copy)
                    os.chmod(debug_copy, 0o644)  # Make readable by all users
                    notify_send(f"Saved debug screenshot to {debug_copy}")
                except Exception as e:
                    debug_log(f"Failed to save debug copy: {e}", args.debug)
                
            # Add small delay to ensure file is written and closed
            time.sleep(0.5)
            
            try:
                debug_log("Sending select command to OCR service", args.debug)
                write_command_to_pipe("select", screenshot_file.name)
            except IOError as e:
                debug_log(f"Failed to communicate with OCR service: {e}", args.debug)
                notify_send(f"Failed to communicate with OCR service: {e}")
                # Cleanup
                os.unlink(screenshot_file.name)
                raise
    except KeyboardInterrupt:
        notify_send("Select freeze cancelled by user")
        sys.exit(1)
    except Exception as e:
        debug_log(f"Unexpected error in select_freeze: {e}\n{traceback.format_exc()}", args.debug)
        notify_send(f"Error: {e}")
        sys.exit(1)


def freeze_screen_directly():
    """
    Take a screenshot of the active monitor and display it fullscreen
    without going through the OCR service. This provides a more direct
    and reliable way to freeze the screen.
    """
    # Get debug state from args or assume False if not available
    debug_enabled = False
    try:
        global args
        if args and hasattr(args, 'debug'):
            debug_enabled = args.debug
    except NameError:
        pass
    
    # Always log to console regardless of debug flag
    print("[freeze_screen_directly] Starting direct screen freeze")
    
    # Create a temporary file for the screenshot
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as screenshot_file:
        screenshot_path = screenshot_file.name
        print(f"[freeze_screen_directly] Creating screenshot at: {screenshot_path}")
        
        # Take screenshot
        try:
            # First try the standard method
            print("[freeze_screen_directly] Taking screenshot of active monitor")
            take_screenshot(screenshot_path, active_monitor=True)
            print("[freeze_screen_directly] Screenshot successful")
        except Exception as e:
            print(f"[freeze_screen_directly] Standard screenshot method failed: {e}")
            # Try alternative methods
            if not try_alternative_screenshot(screenshot_path, debug_enabled):
                notify_send("Failed to create screenshot")
                os.unlink(screenshot_path)
                return
        
        # Verify the screenshot was created successfully
        if not os.path.exists(screenshot_path) or os.path.getsize(screenshot_path) == 0:
            notify_send("Failed to create screenshot")
            os.unlink(screenshot_path)
            return
            
        print(f"[freeze_screen_directly] Screenshot created: {os.path.getsize(screenshot_path)} bytes")
            
        # Try to add text overlay
        temp_image_path = None
        try:
            if shutil.which("convert"):
                print("[freeze_screen_directly] Adding text overlay")
                temp_image_path = f"{screenshot_path}.overlay.png"
                
                # Create a copy with text overlay
                subprocess.run([
                    "convert", screenshot_path,
                    "-fill", "white", "-undercolor", "#00000080",
                    "-gravity", "North", "-pointsize", "24",
                    "-annotate", "+0+30", "SCREEN FROZEN - PRESS ESC OR Q TO EXIT",
                    temp_image_path
                ], check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                
                if os.path.exists(temp_image_path) and os.path.getsize(temp_image_path) > 0:
                    print(f"Using overlay image: {temp_image_path}")
                    display_path = temp_image_path
                else:
                    print("Overlay creation failed, using original image")
                    display_path = screenshot_path
            else:
                print("ImageMagick convert not found, skipping text overlay")
                display_path = screenshot_path
        except Exception as e:
            print(f"[freeze_screen_directly] Failed to add overlay: {e}")
            display_path = screenshot_path
        
        # Display the image
        viewer_cmd = None
        if shutil.which("feh"):
            print("[freeze_screen_directly] Using feh as viewer")
            viewer_cmd = ["feh", "--fullscreen", "--borderless", "--hide-pointer", "--auto-zoom", display_path]
            viewer_exit_msg = "Press q or Escape to unfreeze"
        elif shutil.which("mpv"):
            print("[freeze_screen_directly] Using mpv as viewer")
            viewer_cmd = ["mpv", "--fullscreen", "--keep-open=yes", "--image-display-duration=inf", display_path]
            viewer_exit_msg = "Press q or Escape to unfreeze"
        elif shutil.which("eog"):
            print("[freeze_screen_directly] Using eog as viewer")
            viewer_cmd = ["eog", "--fullscreen", display_path]
            viewer_exit_msg = "Press F11 or Escape to unfreeze"
        elif shutil.which("xdg-open"):
            print("[freeze_screen_directly] Using xdg-open as viewer")
            viewer_cmd = ["xdg-open", display_path]
            viewer_exit_msg = "Close the viewer to unfreeze"
        
        if not viewer_cmd:
            notify_send("No suitable image viewer found")
            os.unlink(screenshot_path)
            if temp_image_path and os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
            return
            
        try:
            print(f"[freeze_screen_directly] Launching viewer: {' '.join(viewer_cmd)}")
            # Use Popen with pipes for stdout/stderr to capture output
            process = subprocess.Popen(
                viewer_cmd, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
            notify_send(f"Screen frozen. {viewer_exit_msg}")
            
            # Wait a moment to see if the process started successfully
            time.sleep(0.5)
            if process.poll() is not None:
                stdout = process.stdout.read().decode('utf-8', errors='ignore')
                stderr = process.stderr.read().decode('utf-8', errors='ignore')
                print(f"Viewer failed to start with code {process.returncode}")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                notify_send("Failed to display frozen screen")
            else:
                print(f"Viewer started with PID {process.pid}")
                # Wait for the process to complete (this will wait until the user closes the viewer)
                process.wait()
                print("Viewer closed")
        except Exception as e:
            print(f"Error displaying frozen screen: {e}\n{traceback.format_exc()}")
            notify_send(f"Error: {e}")
        finally:
            # Clean up temporary files
            try:
                if screenshot_path and os.path.exists(screenshot_path):
                    os.unlink(screenshot_path)
                if temp_image_path and os.path.exists(temp_image_path) and temp_image_path != screenshot_path:
                    os.unlink(temp_image_path)
            except Exception as e:
                print(f"Error cleaning up temporary files: {e}")
    
    # Exit after standalone freeze operation
    sys.exit(0)


def run_select_standalone(image_path=None):
    """Run the select freeze command in standalone mode"""
    if image_path:
        # If an image path is provided, use the original method
        return run_select_freeze(image_path=image_path)
    
    # Otherwise, use the standalone mode by re-executing this script with the special flag
    try:
        print("Starting standalone screen freeze...")
        # Run ourselves as a new process with the standalone flag
        script_path = os.path.abspath(__file__)
        
        log_file = "/tmp/manga_ocr_freeze_log.txt"
        
        # Create a clear marker in the log file
        with open(log_file, "w") as f:
            f.write(f"=== FREEZE STARTED AT {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        
        # Execute in a separate process and redirect stdout/stderr to the log file
        p = subprocess.Popen(
            [sys.executable, script_path, "--standalone-freeze"],
            stdout=open(log_file, "a"),
            stderr=subprocess.STDOUT
        )
        
        # Wait a bit and check if the process is still running
        time.sleep(1)
        if p.poll() is not None:
            # Process ended quickly, probably an error
            print(f"Standalone freeze process exited quickly with code {p.returncode}")
            print("Check log file for details: " + log_file)
            
            # Display the log file contents
            try:
                with open(log_file, "r") as f:
                    log_content = f.read()
                    print("\nLog file contents:")
                    print(log_content)
            except Exception as e:
                print(f"Error reading log file: {e}")
                
            notify_send("Screen freeze failed")
            return False
        
        # Process is running, assume it's working
        print(f"Standalone freeze process started successfully (PID: {p.pid})")
        print(f"Check log file for details: {log_file}")
        
        return True
    except Exception as e:
        print(f"Failed to launch standalone freeze: {e}")
        traceback.print_exc()
        notify_send(f"Error starting screen freeze: {e}")
        return False


def create_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="An OCR tool that uses Transformers.",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode with verbose logging')
    parser.epilog = f"""
Platform: {CURRENT_PLATFORM.name}
You need to run '{prog_name()} download' once after installation.
{prog_name()} home page: https://github.com/Ajatt-Tools/transformers_ocr"""

    subparsers = parser.add_subparsers(title="commands")

    recognize_parser = subparsers.add_parser("recognize", help="OCR a part of the screen.", aliases=["ocr"])
    recognize_parser.add_argument("--image-path", help="Path to image to parse.", metavar="<path>", default=None)
    recognize_parser.set_defaults(func=lambda args: run_ocr("recognize", image_path=args.image_path))

    hold_parser = subparsers.add_parser("hold", help="OCR and hold a part of the screen.")
    hold_parser.add_argument("--image-path", help="Path to image to parse.", metavar="<path>", default=None)
    hold_parser.set_defaults(func=lambda args: run_ocr("hold", image_path=args.image_path))
    
    screenshot_parser = subparsers.add_parser("screenshot", help="Take a screenshot of all screens and copy to clipboard.")
    screenshot_parser.set_defaults(func=lambda _args: run_screenshot_copy(full_screen=True))
    
    select_parser = subparsers.add_parser("select", help="Freeze the current screen by taking a screenshot and displaying it fullscreen.")
    select_parser.add_argument("--image-path", help="Path to image to freeze instead of taking a screenshot.", metavar="<path>", default=None)
    select_parser.set_defaults(func=lambda args: run_select_standalone(image_path=args.image_path))

    download_parser = subparsers.add_parser("download", help="Download OCR files.")
    download_parser.set_defaults(func=lambda _args: download_manga_ocr())

    start_parser = subparsers.add_parser("start", help="Start listening.", aliases=["listen"])
    start_parser.add_argument("--foreground", action="store_true")
    start_parser.set_defaults(func=start_listening)

    stop_parser = subparsers.add_parser("stop", help="Stop listening.")
    stop_parser.set_defaults(func=lambda _args: stop_listening())

    status_parser = subparsers.add_parser("status", help="Print listening status.")
    status_parser.set_defaults(func=lambda _args: print_status())

    restart_parser = subparsers.add_parser("restart", help="Stop listening and start listening.")
    restart_parser.set_defaults(func=lambda _args: restart_listener())

    nuke_parser = subparsers.add_parser("purge", help="Purge all manga-ocr data.", aliases=["nuke"])
    nuke_parser.set_defaults(func=lambda _args: purge_manga_ocr_data())

    return parser


def main():
    prepare_pipe()
    parser = create_args_parser()
    if len(sys.argv) < 2:
        return parser.print_help()
    
    global args  # Make args accessible to debug_log
    args = parser.parse_args()
    
    if args.debug:
        print("[DEBUG] Debug mode enabled")
        
    try:
        if hasattr(args, 'func'):
            if args.func == start_listening:
                # Pass debug flag to MangaOcrWrapper
                global _wrapper
                _wrapper = MangaOcrWrapper(debug=args.debug)
            args.func(args)
    except MissingProgram as ex:
        notify_send(str(ex))
    except ScreenshotCancelled:
        notify_send("Screenshot cancelled.")


# Check if we're being called directly as a standalone script
if __name__ == "__main__" and "--standalone-freeze" in sys.argv:
    # This is a standalone call just to freeze the screen
    
    with open("/tmp/manga_ocr_freeze_log.txt", "a") as log:
        log.write("Starting standalone screen freeze...\n")
        log.write(f"Current directory: {os.getcwd()}\n")
        log.write(f"Python executable: {sys.executable}\n")
        log.write(f"Python path: {sys.path}\n")
        log.write(f"Script path: {__file__}\n")
        log.write(f"Arguments: {sys.argv}\n")
    
    print("Starting standalone screen freeze...")
    
    def notify_send(message):
        """Send a notification"""
        try:
            if shutil.which("notify-send"):
                subprocess.run(["notify-send", "Manga OCR", message], check=False)
            else:
                print(f"NOTIFICATION: {message}")
        except Exception as e:
            print(f"Failed to send notification: {e}")
    
    def take_screenshot(file_path, active_monitor=True, full_screen=False):
        """Take a screenshot and save it to the given file path"""
        print(f"Taking screenshot to {file_path}")
        
        # Default to active monitor if no other option is specified
        if not active_monitor and not full_screen:
            active_monitor = True
            
        if sys.platform == "darwin":  # macOS
            if active_monitor:
                subprocess.run(["screencapture", "-x", file_path], check=True)
            else:  # full_screen
                subprocess.run(["screencapture", "-x", file_path], check=True)
        elif sys.platform == "win32":  # Windows
            # Import only when needed
            try:
                from PIL import ImageGrab
                image = ImageGrab.grab()
                image.save(file_path)
            except Exception as e:
                print(f"Error taking screenshot with PIL: {e}")
                notify_send(f"Error taking screenshot: {e}")
                raise
        else:  # Linux/Unix
            try:
                if shutil.which("maim"):
                    if active_monitor:
                        # Capture active monitor with maim (-i $(xdotool getactivewindow))
                        subprocess.run(["maim", "-u", file_path], check=True)
                    else:  # full_screen
                        subprocess.run(["maim", file_path], check=True)
                elif shutil.which("gnome-screenshot"):
                    subprocess.run(["gnome-screenshot", "-f", file_path], check=True)
                elif shutil.which("scrot"):
                    subprocess.run(["scrot", file_path], check=True)
                elif shutil.which("import"):
                    subprocess.run(["import", "-window", "root", file_path], check=True)
                else:
                    notify_send("No screenshot program found. Please install maim, gnome-screenshot, scrot, or import.")
                    raise FileNotFoundError("No screenshot program found")
            except Exception as e:
                print(f"Error taking screenshot: {e}")
                notify_send(f"Error taking screenshot: {e}")
                raise
    
    def try_alternative_screenshot(file_path):
        """Try alternative methods for taking screenshots"""
        print("Trying alternative screenshot methods...")
        
        methods = [
            # Try full screen methods first
            lambda: subprocess.run(["maim", file_path], check=True),
            lambda: subprocess.run(["gnome-screenshot", "-f", file_path], check=True),
            lambda: subprocess.run(["scrot", file_path], check=True),
            lambda: subprocess.run(["import", "-window", "root", file_path], check=True),
            
            # Try X11 specific methods
            lambda: subprocess.run(["xwd", "-root", "-out", "/tmp/x11_screen.xwd"], check=True) and
                    subprocess.run(["convert", "/tmp/x11_screen.xwd", file_path], check=True),
        ]
        
        for i, method in enumerate(methods):
            try:
                print(f"Trying method {i+1}...")
                method()
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    print(f"Screenshot successful with method {i+1}")
                    return True
            except Exception as e:
                print(f"Method {i+1} failed: {e}")
                continue
                
        return False
    
    # Create a temporary file for the screenshot
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as screenshot_file:
        screenshot_path = screenshot_file.name
        print(f"Creating screenshot at: {screenshot_path}")
        
        # Save a debug copy
        debug_path = "/tmp/manga_ocr_debug_screenshot.png"
        
        try:
            # Take screenshot
            try:
                # First try the standard method
                take_screenshot(screenshot_path, active_monitor=True)
                print("Screenshot successful")
                
                # Save a debug copy
                shutil.copy(screenshot_path, debug_path)
                print(f"Debug copy saved to {debug_path}")
            except Exception as e:
                print(f"Standard screenshot method failed: {e}")
                
                # Try alternative methods
                if not try_alternative_screenshot(screenshot_path):
                    notify_send("Failed to create screenshot")
                    os.unlink(screenshot_path)
                    sys.exit(1)
                
                # Save a debug copy of the alternative screenshot
                shutil.copy(screenshot_path, debug_path)
                print(f"Debug copy of alternative screenshot saved to {debug_path}")
            
            # Verify the screenshot was created successfully
            if not os.path.exists(screenshot_path) or os.path.getsize(screenshot_path) == 0:
                print("Screenshot file is empty or not created")
                notify_send("Failed to create screenshot")
                os.unlink(screenshot_path)
                sys.exit(1)
                
            print(f"Screenshot created: {os.path.getsize(screenshot_path)} bytes")
                
            # Try to add text overlay
            temp_image_path = None
            try:
                if shutil.which("convert"):
                    print("Adding text overlay")
                    temp_image_path = f"{screenshot_path}.overlay.png"
                    
                    # Create a copy with text overlay
                    print("Running convert command...")
                    process = subprocess.run([
                        "convert", screenshot_path,
                        "-fill", "white", "-undercolor", "#00000080",
                        "-gravity", "North", "-pointsize", "24",
                        "-annotate", "+0+30", "SCREEN FROZEN - PRESS ESC OR Q TO EXIT",
                        temp_image_path
                    ], check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                    
                    if os.path.exists(temp_image_path) and os.path.getsize(temp_image_path) > 0:
                        print(f"Using overlay image: {temp_image_path}")
                        display_path = temp_image_path
                        # Save a debug copy of the overlay
                        shutil.copy(temp_image_path, "/tmp/manga_ocr_debug_screenshot_overlay.png")
                    else:
                        print("Overlay creation failed, using original image")
                        display_path = screenshot_path
                else:
                    print("ImageMagick convert not found, skipping text overlay")
                    display_path = screenshot_path
            except Exception as e:
                print(f"Failed to add overlay: {e}\n{traceback.format_exc()}")
                display_path = screenshot_path
            
            # Display the image
            viewer_cmd = None
            if shutil.which("feh"):
                print("Using feh as viewer")
                viewer_cmd = ["feh", "--fullscreen", "--borderless", "--hide-pointer", "--auto-zoom", display_path]
                viewer_exit_msg = "Press q or Escape to unfreeze"
            elif shutil.which("mpv"):
                print("Using mpv as viewer")
                viewer_cmd = ["mpv", "--fullscreen", "--keep-open=yes", "--image-display-duration=inf", display_path]
                viewer_exit_msg = "Press q or Escape to unfreeze"
            elif shutil.which("eog"):
                print("Using eog as viewer")
                viewer_cmd = ["eog", "--fullscreen", display_path]
                viewer_exit_msg = "Press F11 or Escape to unfreeze"
            elif shutil.which("xdg-open"):
                print("Using xdg-open as viewer")
                viewer_cmd = ["xdg-open", display_path]
                viewer_exit_msg = "Close the viewer to unfreeze"
            
            if not viewer_cmd:
                print("No suitable image viewer found")
                notify_send("No suitable image viewer found")
                os.unlink(screenshot_path)
                if temp_image_path and os.path.exists(temp_image_path):
                    os.unlink(temp_image_path)
                sys.exit(1)
                
            try:
                print(f"Launching viewer: {' '.join(viewer_cmd)}")
                # Use Popen with pipes for stdout/stderr to capture output
                process = subprocess.Popen(
                    viewer_cmd, 
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True
                )
                notify_send(f"Screen frozen. {viewer_exit_msg}")
                
                # Wait a moment to see if the process started successfully
                time.sleep(0.5)
                if process.poll() is not None:
                    stdout = process.stdout.read().decode('utf-8', errors='ignore')
                    stderr = process.stderr.read().decode('utf-8', errors='ignore')
                    print(f"Viewer failed to start with code {process.returncode}")
                    print(f"STDOUT: {stdout}")
                    print(f"STDERR: {stderr}")
                    notify_send("Failed to display frozen screen")
                    sys.exit(1)
                else:
                    print(f"Viewer started with PID {process.pid}")
                    # Wait for the process to complete (this will wait until the user closes the viewer)
                    process.wait()
                    print("Viewer closed")
            except Exception as e:
                print(f"Error displaying frozen screen: {e}\n{traceback.format_exc()}")
                notify_send(f"Error: {e}")
                sys.exit(1)
        except KeyboardInterrupt:
            print("Operation cancelled by user")
            notify_send("Screen freeze cancelled")
            sys.exit(0)
        except Exception as e:
            print(f"Unexpected error: {e}\n{traceback.format_exc()}")
            notify_send(f"Error: {e}")
            sys.exit(1)
        finally:
            # Clean up temporary files
            try:
                if screenshot_path and os.path.exists(screenshot_path):
                    os.unlink(screenshot_path)
                if temp_image_path and os.path.exists(temp_image_path) and temp_image_path != screenshot_path:
                    os.unlink(temp_image_path)
            except Exception as e:
                print(f"Error cleaning up temporary files: {e}")
    
    # Exit after standalone freeze operation
    sys.exit(0)


if __name__ == "__main__":
    main()
