/*
 * ADX-12 driver interface - synthetic teaching corpus, not real firmware.
 * Written for the RAG learning path. Register names and error codes match
 * adx12_datasheet.pdf in this folder; every one of them is invented.
 */
#ifndef ADX12_DRIVER_H
#define ADX12_DRIVER_H

#include <stdint.h>

#define ADX12_BASE          (0x40012400u)
#define ADX12_CTRL          (*(volatile uint32_t *)(ADX12_BASE + 0x00u))
#define ADX12_SMPR          (*(volatile uint32_t *)(ADX12_BASE + 0x04u))
#define ADX12_SQR           (*(volatile uint32_t *)(ADX12_BASE + 0x08u))
#define ADX12_DR            (*(volatile uint32_t *)(ADX12_BASE + 0x0Cu))
#define ADX12_ISR           (*(volatile uint32_t *)(ADX12_BASE + 0x10u))

#define ADX12_CTRL_EN       (1u << 0)
#define ADX12_CTRL_START    (1u << 1)
#define ADX12_CTRL_CONT     (1u << 2)
#define ADX12_CTRL_ALIGN    (1u << 3)

#define ADX12_ISR_EOC       (1u << 0)
#define ADX12_ISR_OVR       (1u << 1)

#define ADX12_TIMEOUT_LOOPS (10000u)

typedef enum {
    ADX12_OK = 0,
    ADX12_E_TIMEOUT,    /* no EOC within ADX12_TIMEOUT_LOOPS */
    ADX12_E_OVERRUN,    /* ADX-E17: DR was not read in time */
    ADX12_E_PARAM       /* argument out of range */
} Adx12_StatusType;

typedef struct {
    uint8_t  channel;        /* 0..7 (0..3 on ADX-12B) */
    uint16_t sample_cycles;  /* 1..256 */
    uint8_t  left_align;     /* non-zero sets CTRL.ALIGN */
} Adx12_ConfigType;

void Adx12_Init(const Adx12_ConfigType *config);
Adx12_StatusType Adx12_SetSampleTime(uint16_t sample_cycles);
Adx12_StatusType Adx12_ReadChannel(uint8_t channel, uint16_t *result);
void Adx12_IrqHandler(void);

#endif /* ADX12_DRIVER_H */
