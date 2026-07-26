# SPX-400 SPI DMA Driver Guide

> **Synthetic teaching corpus — not a real datasheet.** Written for the RAG
> learning path (`docs/00_LEARNING_PATH.md`). Every register name, error code
> and errata identifier below is invented.

## 1. Scope

This guide describes the DMA-driven transfer path for the SPX-400 SPI
peripheral in an AUTOSAR-style driver. It covers channel setup, buffer
ownership, and the clock configuration the peripheral shares with the rest of
the chip. Interrupt-driven and polled transfer modes are mentioned only where
they differ from the DMA path.

The driver is assumed to own the peripheral exclusively. Shared-ownership
configurations, where a bootloader and an application both touch the same SPI
instance, are out of scope.

## 2. Transfer modes

Three transfer modes are available, in increasing order of setup cost and
decreasing order of CPU load.

- **Polled** — the CPU writes a byte, spins on the transmit-empty flag, reads
  the received byte. Simple, and acceptable only for configuration traffic.
- **Interrupt-driven** — one interrupt per word. Correct, but the interrupt
  overhead dominates at high clock rates.
- **DMA** — the DMA engine moves data between memory and the peripheral without
  CPU involvement. In an AUTOSAR SPI driver, DMA can transfer SPI data without
  the CPU copying every byte, which is the whole reason to pay the setup cost.

DMA is the only mode that keeps up with continuous streaming at the top of the
peripheral's clock range.

## 3. Clock configuration and the SPI prescaler

The SPI bit clock is derived from the peripheral clock through a prescaler,
exactly as the CAN controller derives its time quantum. The SPX-400 prescaler is
a 3-bit field selecting a power-of-two divisor from 2 through 256.

Selecting the bit clock is a trade-off against signal integrity on the board,
not just a throughput decision. A prescaler that produces a 25 MHz bit clock
will work on a short controlled-impedance trace and fail on a ribbon cable.

Note that the prescaler only applies in master mode. In slave mode the external
master supplies the clock and the prescaler field is ignored — a detail that
regularly confuses driver authors porting a master-mode configuration to a slave
instance.

## 4. Bit timing and clock polarity

Bit timing for SPI is far simpler than for CAN, but the term means the same
thing: where in the bit period the receiver samples the line.

Two fields control it. Clock polarity selects the idle level of the clock line.
Clock phase selects whether data is sampled on the leading or trailing edge.
Together they define the four conventional SPI modes, numbered 0 through 3.

Master and slave must agree on both fields. A polarity mismatch shifts every
sample point by half a bit period, which typically presents as data that is
correct but rotated by one bit — a symptom that looks like a wiring fault and is
not one.

## 5. DMA channel configuration

Two channels are required per full-duplex transfer: one memory-to-peripheral for
transmit, one peripheral-to-memory for receive. Configuring only the transmit
channel is a common shortcut that works until the first read.

Both channels must be configured with the same transfer width as the peripheral
data register. A width mismatch does not raise an error; it silently transfers
the wrong number of bytes per request.

The receive channel should be enabled before the transmit channel. Enabling
transmit first opens a window in which the peripheral can complete a word before
the receive channel is armed, and that word is lost.

## 6. Buffer management and ownership

The driver hands the DMA engine a pointer and a length, and from that moment
until the completion callback the buffer belongs to the DMA engine. Writing to a
buffer that DMA owns is the most common source of intermittent corruption in
this driver, and it is invisible in single-stepping because the debugger halts
the CPU but not the DMA engine.

For an SPI slave, the external master controls the clock and the chip select.
The slave must therefore prepare the transmit buffer before the master starts
clocking data — there is no back-pressure mechanism available to a slave. If the
transmit buffer is not ready in time, underrun occurs and the slave clocks out
whatever stale content the buffer holds.

Buffer alignment must match the DMA transfer width. Misaligned buffers are
rejected at channel configuration time on this peripheral, which is preferable
to the alternative.

## 7. Configuration register reset values

All transfer configuration comes out of reset disabled, so that a partially
initialised driver cannot start a transfer.

The register defaults to 0x00. That selects polled mode, prescaler divisor 2,
clock polarity zero and clock phase zero — SPI mode 0 at the fastest available
bit clock. The combination is legal but almost never what the application wants,
so the driver must write every field explicitly rather than relying on reset
values for the fields it does not care about.

Configuration reset values for this peripheral are documented per field rather
than per register, because the mode and enable fields share a single 32-bit
word.
