"""Small frozen Windows updater; originals and review data remain untouched."""
import json
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from release_bundle import apply_update, checked_manifest


def main():
    root = tk.Tk()
    root.title('Transcript Review Update')
    status = tk.StringVar(value='Close Transcript Review, then select its existing application folder.')
    ttk.Label(root, textvariable=status, wraplength=500, padding=20).pack()
    progress = ttk.Progressbar(root, mode='indeterminate')
    progress.pack(fill='x', padx=20)
    events = queue.Queue()
    busy = False

    def close():
        if busy:
            messagebox.showinfo('Update in progress', 'Please wait for verification and copying to finish. The original installation stays unchanged.')
        else:
            root.destroy()
    root.protocol('WM_DELETE_WINDOW', close)

    def select():
        nonlocal busy
        folder = filedialog.askdirectory(title='Select existing application folder')
        if not folder:
            return
        button.configure(state='disabled')
        busy = True
        progress.start()
        status.set('Verifying files and creating the new installation. The original stays unchanged.')
        def worker():
            try:
                here = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).parent
                spec = json.loads((here / 'update.json').read_text(encoding='utf-8'))
                target = checked_manifest(spec['target'])
                output = Path(folder).parent / f'Transcript-Review-{target["version"]}'
                apply_update(Path(folder), here, output)
                events.put(('ok', str(output)))
            except Exception:  # noqa: BLE001 - sanitized UI boundary; keep source values out of errors
                events.put(('error', 'Update could not be applied. Check the base version, complete extraction, free disk space and that the new version folder does not already exist. The old installation is unchanged.'))
        threading.Thread(target=worker, daemon=True).start()

    button = ttk.Button(root, text='Select installation and update', command=select)
    button.pack(pady=20)
    def poll():
        nonlocal busy
        try:
            kind, value = events.get_nowait()
            busy = False
            progress.stop()
            button.configure(state='normal')
            if kind == 'ok':
                status.set('Update ready. Open the application in: ' + value)
            else:
                status.set(value)
                messagebox.showerror('Update not applied', value)
        except queue.Empty:
            pass
        root.after(100, poll)
    root.after(100, poll)
    root.mainloop()


if __name__ == '__main__':
    main()
