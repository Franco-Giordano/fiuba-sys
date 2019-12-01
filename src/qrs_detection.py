import numpy as np


def qrs_detection(fm):

    candidato_int = np.zeros(fm.size)
    noise_peak_int = np.zeros(fm.size)
    n = 0
    esperar = 0
    rr_avg_2_int = 0

    while n <= fm.size:
        if n == 720:
            aux = fm[0:n]
            thr_signal_int = aux.max()
            sig_level_int = thr_signal_int
            aux = aux[aux.nonzero()]
            thr_noise_int = min(aux)
            noise_level_int = thr_noise_int

        if n > 720 and esperar == 0:
            if fm[n-2] > thr_signal_int:
                candidato_int[n-2] = fm[n-2]
                esperar = 72

                aux, = np.where(candidato_int)
                aux_diff = np.diff(aux)
                principio = max(0, aux_diff.size-8-1)
                if aux_diff.size:
                    rr_avg_1_int = aux_diff[principio:aux_diff.size].mean()
                else:
                    rr_avg_1_int = np.nan

                if rr_avg_2_int == 0:
                    rr_avg_2_int = rr_avg_1_int
                elif aux_diff.size > 8:
                    aux_diff_2, = np.where(np.logical_and(aux_diff > 0.92*rr_avg_2_int, aux_diff < 1.16*rr_avg_2_int))
                    if aux_diff_2.size:
                        principio2 = max(0, aux_diff_2.size-1-8)
                        rr_avg_2_int = aux_diff_2[principio2:aux_diff_2.size].mean()

                rr_missed_limit = rr_avg_2_int*1.66

                if aux_diff.size > 8 and aux_diff[aux_diff.size-1] > rr_missed_limit:
                    fm_aux = fm[(aux[aux.size-2]):(aux[aux.size-1])]
                    aux_fm_aux, = aux[aux.size-2] + 1 + np.where(fm_aux > thr_noise_int)

                    if np.size(aux_fm_aux):
                        candidato_int[n-2] = 0
                        candidato_int[aux_fm_aux[1]-1] = fm[aux_fm_aux[1]-1]
                        n = aux_fm_aux[0] + 1
                        sig_level_int = 0.25*fm[n-2] + 0.75*sig_level_int
                else:
                    sig_level_int = 0.125*fm[n-2] + 0.875*sig_level_int

            else:
                noise_level_int = 0.125*fm[n-2] + 0.875*noise_level_int
                noise_peak_int[n - 2] = 1
            thr_signal_int = noise_level_int + 0.25 *(sig_level_int-noise_level_int)
            thr_noise_int = 0.5*thr_signal_int
        elif esperar > 0:
            esperar = esperar - 1
        n = n + 1

    return candidato_int

