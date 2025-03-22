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
        try:
            # Display image fullscreen to create a freeze effect
            if shutil.which("feh"):
                subprocess.Popen(["feh", "--fullscreen", "--borderless", "--hide-pointer", image_path])
                notify_send("Screen frozen. Press q or Escape to unfreeze.")
            elif shutil.which("eog"):
                subprocess.Popen(["eog", "--fullscreen", image_path])
                notify_send("Screen frozen. Press F11 or Escape to unfreeze.")
            elif shutil.which("mpv"):
                subprocess.Popen(["mpv", "--fullscreen", "--image-display-duration=inf", image_path])
                notify_send("Screen frozen. Press q or Escape to unfreeze.")
            elif shutil.which("xdg-open"):
                subprocess.Popen(["xdg-open", image_path])
                notify_send("Screen frozen. Close the viewer to unfreeze.")
            else:
                notify_send("No image viewer found. Install feh, eog, or mpv for best results.")
        except Exception as e:
            self._debug_log(f"Error displaying image: {e}")
            notify_send(f"Error displaying image: {e}")
            
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
                
            self._wrapper._process_command(command)
            
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
        ensure_listening()
        
        if image_path is not None:
            # Use the provided image directly
            write_command_to_pipe("select", image_path)
            return

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as screenshot_file:
            try:
                # Take screenshot of active monitor to create freeze effect
                take_screenshot(screenshot_file.name, active_monitor=True)
            except subprocess.CalledProcessError as ex:
                raise ScreenshotCancelled() from ex
                
            # Add small delay to ensure file is written
            time.sleep(0.1)
            
            try:
                write_command_to_pipe("select", screenshot_file.name)
            except IOError as e:
                notify_send(f"Failed to communicate with OCR service: {e}")
                # Cleanup
                os.unlink(screenshot_file.name)
                raise
    except KeyboardInterrupt:
        notify_send("Select freeze cancelled by user")
        sys.exit(1)
    
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
    
    select_parser = subparsers.add_parser("select", help="Freeze a selected area of the screen for later OCR.")
    select_parser.add_argument("--image-path", help="Path to image to freeze.", metavar="<path>", default=None)
    select_parser.set_defaults(func=lambda args: run_select_freeze(image_path=args.image_path))

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


if __name__ == "__main__":
    main()
