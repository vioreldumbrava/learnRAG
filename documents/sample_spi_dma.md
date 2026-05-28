# SPI DMA Notes

In an AUTOSAR SPI driver, DMA can transfer SPI data without CPU copying every byte.

For an SPI slave, the external master controls the clock and chip select. Therefore the slave must prepare the transmit buffer before the master starts clocking data.

If the SPI slave transmit buffer is not ready in time, underrun can happen. Double buffering or ring buffering is often used for continuous streaming.
