# CHX-2000 CAN Diagnostics and Errata Reference

> **Synthetic teaching corpus — not a real datasheet.** Written for the RAG
> learning path (`docs/00_LEARNING_PATH.md`). Every register name, error code
> and errata identifier below is invented.

## 1. Purpose

This reference lists the diagnostic codes the CHX-2000 CAN controller reports
through its error status register, plus the silicon errata that a driver author
has to work around. It is the companion to the bit timing guide: timing
problems surface here as codes, and several codes have no meaning until you know
which timing field produced them.

## 2. Error code reference

| Code | Meaning | Usual cause |
|---|---|---|
| ERR080011 | Stuff error | More than five consecutive identical bits; usually a wiring or termination fault. |
| ERR080042 | Form error | A fixed-form bit field held an illegal value; frequently a downstream symptom of a resynchronisation failure. |
| ERR080082 | Phase transition timeout | The controller began the arbitration-to-data phase switch and did not observe a valid secondary sample point within the configured window. |
| ERR080101 | Acknowledge slot empty | No other node acknowledged the frame; the node may be alone on the bus. |
| ERR080204 | Bit error, dominant expected | The node transmitted recessive and read back dominant. |

ERR080082 deserves special attention because it is the only code in the table
that is produced by configuration rather than by the physical layer. Replacing
cables and re-terminating the bus will never clear it. The code points at the
prescaler relationship, not at the wire.

## 3. Errata CHEN0 — classic-mode fallback

**CHEN0** is the errata identifier for the controller's undocumented recovery
behaviour when it cannot align the secondary sample point across a phase switch.

Rather than raising a bus-off condition, the affected silicon abandons the CAN FD
data phase for the remainder of the frame and completes the transmission using
classic CAN framing. The frame is delivered, and no error counter increments, so
functional tests pass. The cost is throughput: classic framing caps the payload
at 8 bytes and runs the whole frame at the arbitration bit rate, which
**roughly halves effective throughput** on a bus configured for a 2 Mbit/s data
phase.

Because the fallback is silent, CHEN0 is usually discovered as a performance
regression rather than as a fault. The diagnostic signature is a rising
ERR080082 count alongside an unchanged bus-off count and an unexplained drop in
frames per second.

The workaround is entirely in configuration: choose prescaler values whose time
quanta divide evenly, as described in the bit timing guide. There is no silicon
revision that removes CHEN0 from this family.

## 4. Bus-off recovery

A node enters bus-off when its transmit error counter passes 255. Recovery is
not automatic on this controller unless the driver has enabled it. The automatic
path waits for 128 occurrences of 11 consecutive recessive bits before
re-enabling the transmitter.

Manual recovery gives the driver a chance to log diagnostics before the counters
are cleared, which is generally what you want in a production ECU. The trade-off
is that a manual recovery path that is never exercised in test will not work in
the field.

## 5. Diagnostic procedure

Work from the code outward rather than from the symptom inward.

- Read the error status register before clearing anything; several codes are
  latched and the clear is destructive.
- Separate configuration codes from physical-layer codes. ERR080082 is
  configuration; the stuff and bit errors are physical.
- Compare the error counter deltas against the frame counter delta. A code that
  rises without the frame counter moving points at initialisation; a code that
  rises in proportion to traffic points at timing or termination.
- Only then reach for an oscilloscope.

## 6. Configuration register reset values

Diagnostic registers reset to a cleared state so that a driver reading them
during initialisation does not act on stale counts from before the reset.

The register defaults to 0x00. Error counters therefore read zero on a fresh
boot, which means a zero counter is not evidence of a healthy bus — it may
simply mean nothing has been transmitted yet.

Configuration reset values for this peripheral are documented per field rather
than per register, because the status and enable fields share a single 32-bit
word.
