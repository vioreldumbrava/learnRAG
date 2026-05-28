"""Background workers so long operations don't freeze the UI.

Each operation runs in its own QThread. Signals carry the result back to the
main thread; never touch UI widgets from inside a worker.
"""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, Signal


class Worker(QObject):
    """Generic worker that runs a callable and emits a result or error.

    Usage:
        worker = Worker(lambda: do_something(...))
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.result.connect(on_result)
        worker.error.connect(on_error)
        thread.start()
    """

    result = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, fn: Callable[[], Any]) -> None:
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        try:
            value = self._fn()
        except Exception as exc:  # surface to the UI
            self.error.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.result.emit(value)
        finally:
            self.finished.emit()


def run_in_thread(
    parent: QObject,
    fn: Callable[[], Any],
    on_result: Callable[[Any], None],
    on_error: Callable[[str], None],
    on_done: Callable[[], None] | None = None,
) -> tuple[QThread, Worker]:
    """Run `fn` in a background QThread and dispatch the result.

    Returns (thread, worker). The worker is also attached as `thread._py_worker`
    so it stays alive even if the caller drops the second tuple element.
    Without that, PySide6 can garbage-collect the Python `Worker` between
    `thread.start()` and the `started` signal firing, leaving the operation
    to silently no-op.
    """

    thread = QThread(parent)
    worker = Worker(fn)
    worker.moveToThread(thread)

    thread.started.connect(worker.run)
    worker.result.connect(on_result)
    worker.error.connect(on_error)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    if on_done is not None:
        worker.finished.connect(on_done)

    # Pin Python-side references for the duration of the run.
    thread._py_worker = worker  # type: ignore[attr-defined]
    thread.finished.connect(lambda: setattr(thread, "_py_worker", None))

    thread.start()
    return thread, worker
