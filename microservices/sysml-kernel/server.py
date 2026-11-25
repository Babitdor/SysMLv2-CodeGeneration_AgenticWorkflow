"""Lightweight HTTP server for headless SysML v2 → Graphviz PNG rendering.

Uses Python's built-in ``http.server`` (zero dependencies beyond stdlib).
The server accepts POST requests with SysML v2 textual content and returns
a base64-encoded PNG data URL rendered via the jupyter-sysml-kernel.

Designed to run as a Docker sidecar container so the main backend doesn't
need Java 21, Graphviz, or the Jupyter kernel installed.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import subprocess
import tempfile
from time import monotonic
import uuid
import queue
import select
import socket
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("sysml-kernel-server")

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8888"))
KERNEL_TIMEOUT = int(os.environ.get("KERNEL_TIMEOUT", "120"))
KERNEL_WORKERS = ThreadPoolExecutor(max_workers=2)
KERNEL_SLOTS = threading.BoundedSemaphore(10)  # Two active kernels and eight queued.
REQUEST_TIMEOUT = 300

# Regex to extract base64 PNG from kernel cell output
_PNG_DATA_RE = re.compile(r"data:image/png;base64,([A-Za-z0-9+/=]+)")


def _viz_command(element_name: str, orientation: str, compartments: str) -> str:
    """Validate the direct HTTP boundary before constructing a kernel command."""
    if orientation not in {"TB", "LR"} or compartments not in {"default", "compact", "tree"}:
        raise ValueError("Unsupported diagram orientation or compartment style")
    target = element_name.strip()
    identifier = r"(?:[A-Za-z_]\w*|'[^'\r\n\x00]+'|\"[^\"\r\n\x00]+\")"
    if not re.fullmatch(rf"{identifier}(?:::{identifier})*", target):
        if re.fullmatch(r"[\w -]+", target) and not target.startswith("-"):
            target = f"'{target}'"
        else:
            raise ValueError("Use a qualified element name, for example Package::'Part Name'")
    style = {"default": "", "compact": " --style=COMPMOST", "tree": " --style=COMPTREE"}[compartments]
    return f"%viz --style={orientation}{style} {target}"


def _next_message(client, deadline: float, cancel_event=None):
    while True:
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Kernel request cancelled")
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise queue.Empty
        try:
            return client.get_iopub_msg(timeout=min(remaining, 1) if cancel_event is not None else remaining)
        except queue.Empty:
            if cancel_event is None:
                raise


def _render(sysml_text: str, element_name: str = "", orientation: str = "TB", compartments: str = "default", *, cancel_event=None) -> dict:
    """Render SysML content to a Graphviz diagram.

    Launches the SysML kernel programmatically via jupyter_client, executes the
    SysML v2 code block, and captures the inline SVG/PNG rendering results.
    """
    import jupyter_client
    import queue

    # Auto-detect element name if not provided
    if not element_name:
        pkg_match = re.search(r"\bpackage\s+(\w+)", sysml_text)
        if pkg_match:
            element_name = pkg_match.group(1)
        else:
            # Check for package with quotes (e.g. package '18-Use Case')
            pkg_match = re.search(r"\bpackage\s+'([^']+)'", sysml_text)
            if pkg_match:
                element_name = pkg_match.group(1)
            else:
                return {
                    "success": False,
                    "message": "Could not auto-detect a package name. "
                    "Provide an explicit element_name.",
                }

    try:
        viz_command = _viz_command(element_name, orientation, compartments)
    except (ValueError, TypeError, AttributeError) as exc:
        return {"success": False, "message": str(exc)}

    try:
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Kernel request cancelled before startup")
        km, kc = jupyter_client.manager.start_new_kernel(kernel_name='sysml')
    except Exception as exc:
        return {
            "success": False,
            "message": f"Failed to start SysML Jupyter kernel sidecar: {exc}"
        }

    try:
        # Step 1: Send SysML text
        load_id = kc.execute(sysml_text)
        
        # Wait for kernel to execute and go back to idle
        errors = []
        loaded = False
        deadline = monotonic() + KERNEL_TIMEOUT
        while True:
            try:
                msg = _next_message(kc, deadline, cancel_event)
                if msg.get('parent_header', {}).get('msg_id') != load_id:
                    continue
                msg_type = msg['header']['msg_type']
                content = msg['content']
                if msg_type == 'stream':
                    text = content.get('text', '')
                    if content.get('name') == 'stderr':
                        errors.append(text)
                elif msg_type == 'error':
                    errors.append("\n".join(content.get('traceback', [])))
                elif msg_type == 'status' and content.get('execution_state') == 'idle':
                    loaded = True
                    break
            except queue.Empty:
                break

        if not loaded:
            return {"success": False, "message": "Rendering timed out before model compilation completed."}
        if errors:
            return {"success": False, "message": "\n".join(errors)}

        # Step 2: Send viz command
        # Quote element name if it has spaces/hyphens and is not already quoted
        msg_id = kc.execute(viz_command)
        
        svg_data = None
        png_data = None
        completed = False
        deadline = monotonic() + KERNEL_TIMEOUT
        
        while True:
            try:
                msg = _next_message(kc, deadline, cancel_event)
                if msg.get('parent_header', {}).get('msg_id') != msg_id:
                    continue
                msg_type = msg['header']['msg_type']
                content = msg['content']
                
                if msg_type in ('execute_result', 'display_data'):
                    data = content.get('data', {})
                    if 'image/svg+xml' in data:
                        svg_data = data['image/svg+xml']
                    elif 'image/png' in data:
                        png_data = data['image/png']
                    elif 'text/plain' in data:
                        text = data['text/plain']
                        if "data:image/png;base64," in text:
                            png_data = text.split("data:image/png;base64,")[1].strip()
                elif msg_type == 'stream':
                    text = content.get('text', '')
                    if content.get('name') == 'stderr':
                        errors.append(text)
                elif msg_type == 'error':
                    errors.append("\n".join(content.get('traceback', [])))
                elif msg_type == 'status' and content.get('execution_state') == 'idle':
                    if msg.get('parent_header', {}).get('msg_id') == msg_id:
                        completed = True
                        break
            except queue.Empty:
                break

        if not completed:
            return {"success": False, "message": "Rendering timed out before visualization completed. Try a smaller package or element."}
        if errors:
            return {"success": False, "message": "\n".join(errors)}

        if svg_data:
            b64_svg = base64.b64encode(svg_data.encode('utf-8')).decode('utf-8')
            return {
                "success": True,
                "data_url": f"data:image/svg+xml;base64,{b64_svg}"
            }
        elif png_data:
            png_clean = "".join(png_data.split())
            return {
                "success": True,
                "data_url": f"data:image/png;base64,{png_clean}"
            }
        else:
            # Filter out default Intercax warning if other actual errors are present
            filtered_errors = [e for e in errors if "Project doesn't exist" not in e and e.strip()]
            err_msg = "\n".join(filtered_errors) if filtered_errors else "\n".join(errors)
            if not err_msg.strip():
                err_msg = f"No visualization data returned by kernel for element {element_name}."
            return {
                "success": False,
                "message": err_msg
            }
            
    except Exception as exc:
        return {
            "success": False,
            "message": f"Rendering error: {exc}"
        }
    finally:
        try:
            try:
                kc.stop_channels()
            finally:
                km.shutdown_kernel(now=True)
        except Exception:
            pass


def _validate(sysml_text: str, *, cancel_event=None) -> dict:
    """Validate SysML content using the official OMG compiler/kernel.

    Launches the SysML kernel programmatically, executes the code, and
    collects all syntax/semantic compilation errors.
    """
    import jupyter_client
    import queue

    try:
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Kernel request cancelled before startup")
        km, kc = jupyter_client.manager.start_new_kernel(kernel_name='sysml')
    except Exception as exc:
        return {
            "success": False,
            "errors": [f"Failed to start SysML Jupyter kernel sidecar: {exc}"],
            "completed": False,
        }

    try:
        # Send SysML text for execution
        message_id = kc.execute(sysml_text)
        
        errors = []
        completed = False
        deadline = monotonic() + KERNEL_TIMEOUT
        while True:
            try:
                msg = _next_message(kc, deadline, cancel_event)
                # Startup/other-request idle messages cannot complete this cell.
                if msg.get('parent_header', {}).get('msg_id') != message_id:
                    continue
                msg_type = msg['header']['msg_type']
                content = msg['content']
                if msg_type == 'stream':
                    text = content.get('text', '')
                    if content.get('name') == 'stderr':
                        errors.extend([line.strip() for line in text.splitlines() if line.strip()])
                elif msg_type == 'error':
                    errors.extend(content.get('traceback') or [
                        f"{content.get('ename', 'Compiler error')}: {content.get('evalue', '')}"
                    ])
                elif msg_type == 'status' and content.get('execution_state') == 'idle':
                    completed = True
                    break
            except queue.Empty:
                break

        # Suppress known runtime warnings, never arbitrary compiler errors.
        filtered_errors = [
            e for e in errors 
            if "A terminally deprecated method" not in e
            and not e.startswith("WARNING")
            and not e.startswith("log4j:")
        ]
        if not completed:
            filtered_errors.append("Validation timed out before compilation completed.")
        
        success = completed and len(filtered_errors) == 0
        return {
            "success": success,
            "completed": completed,
            "errors": filtered_errors
        }
            
    except Exception as exc:
        return {
            "success": False,
            "errors": [f"Validation error: {exc}"],
            "completed": False,
        }
    finally:
        try:
            try:
                kc.stop_channels()
            finally:
                km.shutdown_kernel(now=True)
        except Exception:
            pass


class RenderHandler(BaseHTTPRequestHandler):
    """HTTP handler for POST /render and POST /validate."""

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def do_POST(self) -> None:
        if self.path not in ("/render", "/validate"):
            self._json_response(404, {"success": False, "message": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            if length < 1 or length > 2_000_000:
                self._json_response(413, {"success": False, "message": "Request exceeds the 2 MB limit."})
                return
            body = self.rfile.read(length)
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            self._json_response(400, {
                "success": False,
                "message": f"Invalid JSON: {exc}",
            })
            return

        if not isinstance(data, dict):
            self._json_response(400, {"success": False, "message": "Expected a JSON object."})
            return
        sysml_text = data.get("sysml_text", "")

        if not isinstance(sysml_text, str) or not sysml_text:
            self._json_response(400, {
                "success": False,
                "message": "Missing required field: sysml_text",
            })
            return

        if self.path == "/render":
            element_name = data.get("element_name", "")
            logger.info(
                "Rendering %d bytes of SysML (element=%s)",
                len(sysml_text), element_name or "(auto)",
            )
            result = self._kernel_result(_render, sysml_text, element_name,
                                         data.get("orientation", "TB"),
                                         data.get("compartments", "default"))
            if result is None:
                return
            status = 200 if result.get("success") else 422
            self._json_response(status, result)
        else:
            logger.info("Validating %d bytes of SysML", len(sysml_text))
            result = self._kernel_result(_validate, sysml_text)
            if result is None:
                return
            status = 200
            self._json_response(status, result)

    def _kernel_result(self, operation, *args):
        if not KERNEL_SLOTS.acquire(blocking=False):
            self._json_response(503, {"success": False, "completed": False,
                                      "message": "Kernel queue is full; retry later.",
                                      "errors": ["Kernel queue is full; compilation did not run."]})
            return None
        cancelled = threading.Event()
        deadline = monotonic() + REQUEST_TIMEOUT
        try:
            future = KERNEL_WORKERS.submit(operation, *args, cancel_event=cancelled)
        except Exception:
            KERNEL_SLOTS.release()
            self._json_response(503, {"success": False, "completed": False, "message": "Kernel workers unavailable.", "errors": ["Compilation did not run."]})
            return None
        future.add_done_callback(lambda _: KERNEL_SLOTS.release())
        try:
            while True:
                try:
                    return future.result(timeout=0.25)
                except FutureTimeout:
                    if future.done():
                        self._json_response(500, {"success": False, "completed": False, "message": "Kernel worker failed.", "errors": ["Kernel worker failed before completion."]})
                        return None
                    if monotonic() >= deadline:
                        self._json_response(504, {"success": False, "completed": False,
                                                  "message": "Kernel request expired; retry a smaller model.",
                                                  "errors": ["Kernel request expired before completion."]})
                        return None
                    try:
                        readable, _, _ = select.select([self.connection], [], [], 0)
                        if readable and not self.connection.recv(1, socket.MSG_PEEK):
                            return None
                    except (OSError, ValueError):
                        return None
                except Exception:
                    logger.exception("Kernel worker failed")
                    self._json_response(500, {"success": False, "completed": False, "message": "Kernel worker failed.", "errors": ["Kernel worker failed before completion."]})
                    return None
        finally:
            if not future.done():
                cancelled.set()
                future.cancel()

    def _json_response(self, status: int, data: dict) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
        except (ConnectionError, TimeoutError):
            logger.info("Client disconnected before the kernel response was delivered")

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, fmt: str, *args: object) -> None:
        logger.info(fmt, *args)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), RenderHandler)
    logger.info("SysML kernel server listening on %s:%s", HOST, PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.server_close()


if __name__ == "__main__":
    main()
