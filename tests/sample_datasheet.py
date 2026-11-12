"""Source of `documents/ADC/adx12_datasheet.pdf`.

The sample datasheet is generated from the page list below so that its text is
reviewable in a diff and reproducible byte for byte. To change the PDF, edit
`PAGES` and run:

    python -m tests.sample_datasheet

`tests/test_sample_corpus.py` fails if the committed PDF and this source drift
apart. Like the rest of `documents/`, the content is invented teaching
material: no register, code or timing value here describes a real part.
"""

from __future__ import annotations

from pathlib import Path

from tests.pdf_fixture import make_pdf


TARGET = Path(__file__).resolve().parents[1] / "documents" / "ADC" / "adx12_datasheet.pdf"

PAGES: list[list] = [
    [
        "# ADX-12 12-bit SAR ADC - Reference Sheet",
        "Synthetic teaching corpus - not a real datasheet. Written for the RAG learning path.",
        "Every register name, error code and timing value on these pages is invented.",
        "",
        "# 1. Overview",
        "The ADX-12 is a 12-bit successive-approximation converter with eight analog inputs.",
        "One conversion takes 14 clock cycles plus the programmed sample time.",
        "The converter is controlled through five memory-mapped registers (section 2).",
        "Results are right-aligned by default; set CTRL.ALIGN for left alignment.",
        "",
        "# 1.1 Ordering codes",
        [
            ["Code", "Package", "Channels", "Max clock"],
            ["ADX-12A", "QFN-24", "8", "14 MHz"],
            ["ADX-12B", "TSSOP-20", "4", "14 MHz"],
            ["ADX-12Q", "QFN-24", "8", "36 MHz"],
        ],
    ],
    [
        "# 2. Register map",
        "Table 2 lists the registers. Offsets are relative to the peripheral base address.",
        [
            ["Register", "Offset", "Reset", "Access", "Purpose"],
            ["CTRL", "0x00", "0x0000", "RW", "Enable, start, mode"],
            ["SMPR", "0x04", "0x0003", "RW", "Sample time in clock cycles"],
            ["SQR", "0x08", "0x0000", "RW", "Channel sequence"],
            ["DR", "0x0C", "0x0000", "RO", "Conversion result"],
            ["ISR", "0x10", "0x0000", "RW1C", "Status and error flags"],
        ],
        "Writes to SMPR and SQR are ignored while CTRL.EN is set.",
        "Clear CTRL.EN, write the register, then set CTRL.EN again.",
        "Reading DR clears the EOC flag in ISR.",
    ],
    [
        "# 3. CTRL bit fields",
        [
            ["Field", "Bits", "Meaning"],
            ["EN", "0", "Converter enable"],
            ["START", "1", "Start one conversion; self-clearing"],
            ["CONT", "2", "Continuous conversion"],
            ["ALIGN", "3", "Left-align the result in DR"],
            ["RES", "5:4", "Resolution: 0=12, 1=10, 2=8 bit"],
        ],
        "START is ignored unless EN has been set for at least 2 clock cycles.",
        "",
        "# 3.1 Sample time",
        "SMPR holds the sample time minus one, so the reset value 0x0003 means 4 cycles.",
        "Total conversion time in clock cycles = 14 + SMPR + 1.",
        "High-impedance sources need a longer sample time; see section 5.",
    ],
    [
        "# 4. Timing",
        [
            ["Parameter", "Min", "Typ", "Max", "Unit"],
            ["f_clk (ADX-12A/B)", "0.5", "", "14", "MHz"],
            ["f_clk (ADX-12Q)", "0.5", "", "36", "MHz"],
            ["t_conv", "", "14", "", "cycles"],
            ["t_wake", "", "20", "", "us"],
            ["t_sample", "1", "4", "256", "cycles"],
        ],
        "After setting CTRL.EN wait t_wake before the first START.",
        "",
        "# 5. Source impedance",
        "Sources above 10 kOhm need at least 16 sample cycles for 12-bit accuracy.",
        "Sources above 50 kOhm need at least 64 sample cycles.",
    ],
    [
        "# 6. Status flags and error codes",
        [
            ["Flag", "Bit", "Code", "Meaning"],
            ["EOC", "0", "", "End of conversion; result is in DR"],
            ["OVR", "1", "ADX-E17", "Overrun: DR was not read in time"],
            ["CLKE", "2", "ADX-E21", "Clock below f_clk minimum"],
            ["SEQE", "3", "ADX-E30", "SQR selects a channel that does not exist"],
        ],
        "Flags are cleared by writing 1 to them (RW1C). Writing 0 has no effect.",
        "ADX-E17 does not stop the converter; the newest result overwrites DR.",
        "ADX-E30 is raised on ADX-12B when SQR selects channels 4 to 7.",
        "",
        "# 6.1 Recovery",
        "After ADX-E21 clear CTRL.EN, restore the clock, and wait t_wake again.",
    ],
]


if __name__ == "__main__":
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    print(f"wrote {make_pdf(TARGET, PAGES)}")
