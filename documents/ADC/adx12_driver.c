/*
 * ADX-12 driver - synthetic teaching corpus, not real firmware.
 * Written for the RAG learning path. It implements the rules stated in
 * adx12_datasheet.pdf (sections 2, 3 and 6); every name is invented.
 */
#include "adx12_driver.h"

static volatile uint32_t Adx12_OverrunCount = 0u;

/* Busy-wait for at least t_wake (20 us) after setting CTRL.EN. */
static void Adx12_Lcl_WaitWake(void)
{
    volatile uint32_t i;

    for (i = 0u; i < 400u; i++) {
        /* calibrated for a 16 MHz core clock */
    }
}

/*
 * Bring the converter up: program the sample time and alignment while the
 * converter is disabled, then enable it and wait t_wake.
 */
void Adx12_Init(const Adx12_ConfigType *config)
{
    ADX12_CTRL = 0u;                                    /* EN = 0: SMPR and SQR are writable */
    ADX12_SMPR = (uint32_t)config->sample_cycles - 1u;  /* the register holds cycles - 1 */
    ADX12_SQR  = (uint32_t)config->channel;
    if (config->left_align != 0u) {
        ADX12_CTRL |= ADX12_CTRL_ALIGN;
    }
    ADX12_CTRL |= ADX12_CTRL_EN;
    Adx12_Lcl_WaitWake();
}

/*
 * Change the sample time at run time. SMPR ignores writes while CTRL.EN is
 * set, so the converter is disabled first and re-enabled afterwards.
 */
Adx12_StatusType Adx12_SetSampleTime(uint16_t sample_cycles)
{
    uint32_t ctrl;

    if ((sample_cycles < 1u) || (sample_cycles > 256u)) {
        return ADX12_E_PARAM;
    }
    ctrl = ADX12_CTRL;
    ADX12_CTRL = ctrl & ~ADX12_CTRL_EN;
    ADX12_SMPR = (uint32_t)sample_cycles - 1u;
    ADX12_CTRL = ctrl;
    if ((ctrl & ADX12_CTRL_EN) != 0u) {
        Adx12_Lcl_WaitWake();
    }
    return ADX12_OK;
}

/* Convert one channel and return the 12-bit result. */
Adx12_StatusType Adx12_ReadChannel(uint8_t channel, uint16_t *result)
{
    uint32_t loops = ADX12_TIMEOUT_LOOPS;

    if ((channel > 7u) || (result == 0)) {
        return ADX12_E_PARAM;
    }
    ADX12_CTRL &= ~ADX12_CTRL_EN;
    ADX12_SQR = (uint32_t)channel;
    ADX12_CTRL |= ADX12_CTRL_EN;
    Adx12_Lcl_WaitWake();
    ADX12_CTRL |= ADX12_CTRL_START;

    while ((ADX12_ISR & ADX12_ISR_EOC) == 0u) {
        if (loops-- == 0u) {
            return ADX12_E_TIMEOUT;
        }
    }
    *result = (uint16_t)(ADX12_DR & 0x0FFFu);  /* reading DR clears EOC */
    if ((ADX12_ISR & ADX12_ISR_OVR) != 0u) {
        ADX12_ISR = ADX12_ISR_OVR;              /* RW1C: write 1 to clear ADX-E17 */
        return ADX12_E_OVERRUN;
    }
    return ADX12_OK;
}

/* Interrupt handler: count overruns (ADX-E17) and clear the flag. */
void Adx12_IrqHandler(void)
{
    if ((ADX12_ISR & ADX12_ISR_OVR) != 0u) {
        Adx12_OverrunCount++;
        ADX12_ISR = ADX12_ISR_OVR;
    }
}
