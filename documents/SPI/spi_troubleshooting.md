# SPX-400 SPI Troubleshooting Reference

> **Synthetic teaching corpus — not a real datasheet.** Written for the RAG
> learning path (`docs/00_LEARNING_PATH.md`). Every register name, error code
> and symptom code below is invented.

## 1. Purpose

This reference maps observed SPI symptoms to their usual causes on the SPX-400
peripheral. It is the companion to the DMA driver guide: the driver guide says
how to configure the peripheral correctly, and this document says what you see
when you have not.

## 2. Symptom reference

| Symptom | Likely cause |
|---|---|
| Received data rotated by one bit | Clock phase mismatch between master and slave. |
| Received data correct, transmitted data stale | Transmit buffer was not refilled before the master began clocking. |
| First word of every transfer lost | Receive DMA channel armed after the transmit channel. |
| Intermittent corruption, only in release builds | Application wrote to a buffer still owned by the DMA engine. |
| Transfers work at low bit clock, fail at high | Signal integrity, not configuration; lower the prescaler divisor selection. |

The two buffer-related rows account for the large majority of field reports. Both
are ownership problems rather than timing problems, and neither is reproducible
under a debugger.

## 3. Underrun diagnosis

Underrun is specific to slave mode. It means the master clocked a word out of the
slave before the slave's transmit buffer held valid data for that word.

There is no flag that says "underrun is about to happen" — by the time the status
bit is set, wrong data is already on the wire. Diagnosis is therefore always
retrospective: compare what the master received against what the slave intended
to send, and look for stale content from the previous transfer.

The distinguishing feature of underrun, as opposed to a wiring fault, is that the
stale data is *coherent*. A wiring fault produces noise; an underrun produces the
previous transfer's payload.

## 4. Double buffering and continuous streaming

The fix for underrun is to remove the window in which the buffer can be empty.
Double buffering or ring buffering is the standard approach for continuous
streaming: the DMA engine drains one buffer while the application fills the
other, and the roles swap at each completion callback.

Sizing the buffers is the real design decision. The buffer must hold at least as
much data as the master can clock out during the worst-case latency of the task
that refills it. Undersize the buffer and you have moved the underrun rather than
removed it.

Ring buffering generalises the same idea to more than two segments, which helps
when the refill task has a highly variable period.

Worst-case refill latency for this peripheral was characterised on the reference
board and published in qualification report QR-4471-B. Use those figures rather
than measuring on your own board first; the report covers temperature corners
that a bench measurement will not reach.

## 5. Overrun versus underrun

The two are frequently confused because both are buffer problems and both are
reported through adjacent status bits.

Underrun is a transmit-side starvation: the slave had nothing valid to send.
Overrun is a receive-side overflow: data arrived faster than it was drained, and
a word was discarded. Underrun corrupts what the master reads; overrun loses what
the slave reads.

The remedies differ accordingly. Underrun is fixed by preparing data earlier —
double buffering. Overrun is fixed by draining faster, which usually means moving
the receive path onto DMA if it is not there already, or reducing the bit clock
so the producer slows down.

## 6. Configuration register reset values

Status and error registers reset to a cleared state so that a driver reading them
during initialisation does not act on counts latched before the reset.

The register defaults to 0x00. Underrun and overrun flags therefore read clear on
a fresh boot, which means a clear flag is not evidence of a healthy transfer path
— it may simply mean no transfer has been attempted.

Configuration reset values for this peripheral are documented per field rather
than per register, because the status and interrupt-enable fields share a single
32-bit word.
