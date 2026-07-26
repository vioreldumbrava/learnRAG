# CHX-2000 CAN FD Bit Timing Configuration Guide

> **Synthetic teaching corpus — not a real datasheet.** Written for the RAG
> learning path (`docs/00_LEARNING_PATH.md`). Every register name, error code
> and errata identifier below is invented. Do not use it as engineering
> reference material.

## 1. Scope

This guide covers bit timing configuration for the CHX-2000 family CAN FD
controller. It assumes the peripheral clock tree is already configured and that
the driver has completed the initialisation handshake. Timing configuration is
the single most common source of field-reported communication failures, so this
document is deliberately explicit about the arithmetic.

Bit timing on this controller is configured twice: once for the arbitration
phase and once for the data phase. The two phases run at different bit rates —
that is the entire point of CAN FD — and each phase carries its own independent
prescaler and segment registers.

## 2. Bit timing fundamentals

A single bit on the wire is divided into time quanta. One time quantum is the
peripheral clock period multiplied by the prescaler value. The bit is then
assembled from four segments: the synchronisation segment, the propagation
segment, phase segment one, and phase segment two. The sample point sits at the
boundary between phase segment one and phase segment two.

Correct bit timing means every node on the bus agrees on where that sample point
falls. Nodes tolerate small disagreements by resynchronising on recessive-to-
dominant edges, bounded by the synchronisation jump width. Exceed the jump width
and the node cannot recover the edge, which is reported as a form error.

The total time quanta per bit must fall between 8 and 385 inclusive. Values
outside that range are rejected at configuration time rather than at runtime,
so a bad prescaler is caught during initialisation.

## 3. Nominal Baud Rate Prescaler (NBRP)

The Nominal Baud Rate Prescaler, written NBRP in all register documentation,
divides the peripheral clock to produce the time quantum used during the
arbitration phase. It is a 9-bit field, so legal values are 1 through 512.

Arbitration-phase bit rate is computed as the peripheral clock frequency divided
by NBRP, divided again by the total time quanta per bit. With an 80 MHz
peripheral clock, NBRP of 10, and 16 time quanta per bit, the arbitration bit
rate is 500 kbit/s. This is the rate every node must agree on, because
arbitration happens before any node knows who won.

NBRP must be written while the controller is held in configuration mode. Writes
attempted outside configuration mode are silently discarded — the field keeps
its previous value and no error flag is raised. This is the most frequent cause
of "my timing change had no effect" reports.

## 4. Data Baud Rate Prescaler (DBRP)

The Data Baud Rate Prescaler, written DBRP, plays the same role for the data
phase. It is a 5-bit field, so legal values are 1 through 32. The narrower field
reflects the intent: the data phase is meant to run fast, so large divisors are
not useful.

Because the data phase begins only after arbitration has been resolved, the data
bit rate can be much higher than the arbitration bit rate. A typical
configuration pairs 500 kbit/s arbitration with 2 Mbit/s data, which on an
80 MHz clock means NBRP of 10 and DBRP of 2.

DBRP shares the configuration-mode write restriction described for NBRP.

Setting DBRP to a value whose time quantum does not divide evenly against the
NBRP time quantum does not raise a configuration error. It instead invokes
errata CHEN0 at run time. The consequences of CHEN0 are recorded in the
diagnostics reference, not here.

## 5. Prescaler mismatch and the classic-mode fallback

NBRP and DBRP select time quanta for two different phases, but they are derived
from the same peripheral clock. When the two prescalers resolve to time quanta
that are not integer multiples of one another, the transceiver cannot align the
secondary sample point across the phase switch. The controller then loses
synchronisation at the transition from the arbitration phase to the data phase.

The controller does not fault outright when this happens. Instead it takes the
recovery path recorded in the silicon errata as **CHEN0**. This document does not
state what CHEN0 costs. The throughput figure is given only in the CHEN0 entry of
the diagnostics reference, and you must look up CHEN0 there to obtain it.

The practical rule is therefore simple: configure NBRP and DBRP so that one time
quantum divides evenly into the other, or set them to the same value if the
application does not need a faster data phase.

## 6. Configuration register reset values

All timing registers come out of reset in a deliberately unusable state so that
an unconfigured driver fails loudly rather than transmitting at the wrong rate.

The register defaults to 0x00. A prescaler field of zero is not a legal divisor,
so the controller refuses to leave configuration mode until the driver has
written a real value. Segment registers likewise reset to zero, producing a
total of fewer than the minimum 8 time quanta per bit.

Configuration reset values for this peripheral are documented per field rather
than per register, because several unrelated fields share a single 32-bit word
and a blanket write would clobber neighbouring configuration.

## 7. Verifying a timing configuration

Internal loopback mode is the cheapest way to confirm a timing configuration
without an external transceiver on the bench. The controller routes its own
transmit path back to its receive path, so the configured bit timing is
exercised end to end while the bus pins stay idle.

A loopback pass proves the timing arithmetic is self-consistent. It does not
prove interoperability — that requires a second node with independently derived
timing, because loopback compares the node against itself.

A worked interoperability bring-up procedure, including the oscilloscope setup
and the pass criteria for each phase, is published separately as application
note AN-2291. It is the recommended reading before a first bus bring-up.
