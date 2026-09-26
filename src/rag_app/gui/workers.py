"""Run blocking operations in Qt threads and marshal callbacks to the UI."""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, Signal, Slot


class Worker(QThread):
    """One blocking callable; the QThread itself remains owned by the UI."""

    result = Signal(object)
    error = Signal(str)

    def __init__(self, fn: Callable[[], Any], parent: QObject) -> None:
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        try:
            value = self._fn()
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.result.emit(value)


class _Dispatcher(QObject):
    """Explicit receiver whose slots always execute in the UI thread."""

    def __init__(self, on_result, on_error, on_done, parent):
        super().__init__(parent)
        self.on_result, self.on_error, self.on_done = on_result, on_error, on_done

    @Slot(object)
    def result(self, value):
        self.on_result(value)

    @Slot(str)
    def error(self, message):
        self.on_error(message)

    @Slot()
    def done(self):
        if self.on_done:
            self.on_done()


def run_in_thread(
    parent: QObject,
    fn: Callable[[], Any],
    on_result: Callable[[Any], None],
    on_error: Callable[[str], None],
    on_done: Callable[[], None] | None = None,
) -> tuple[QThread, Worker]:
    """Start a callable and return its thread; callbacks run on the UI thread."""
    thread = Worker(fn, parent)
    dispatcher = _Dispatcher(on_result, on_error, on_done, parent)
    thread.result.connect(dispatcher.result)
    thread.error.connect(dispatcher.error)
    thread.finished.connect(dispatcher.done)
    thread.finished.connect(dispatcher.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread._dispatcher = dispatcher  # keep Python callback wrapper alive
    thread.start()
    return thread, thread
