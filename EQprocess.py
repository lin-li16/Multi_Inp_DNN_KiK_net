import numpy as np
import scipy
from scipy import signal
import os
from accProcess import getvel_dsp
import matlab.engine
matlabeng = matlab.engine.start_matlab()   #启动matlab


def EQprocess(acc, dt):
    dt = float(dt)
    t = np.linspace(dt, dt * len(acc), len(acc))
    loc_uncer = 0.2

    # Remove mean
    acc = acc - np.mean(acc)

    # Copy acc
    acc_copy1 = np.copy(acc)
    acc_copy2 = np.copy(acc)

    # Remove linear trend
    coef1 = np.polyfit(t, acc_copy1, 1)
    acc_fit = coef1[0] * t + coef1[1]
    acc_copy1 = acc_copy1 - acc_fit

    # Acausal bandpass filter
    sos = signal.butter(4, [0.1, 20], 'bandpass', fs=int(1 / dt), output='sos')
    acc_filter = signal.sosfilt(sos, acc_copy1)

    # Find event onset
    loc, _ = matlabeng.PphasePicker(matlab.double(acc_filter.tolist()), dt, 'sm', nargout=2)

    # Initial baseline correction: remove pre-event mean
    acc_copy2 = acc_copy2 - np.mean(acc_copy2[:int((loc - loc_uncer) / dt)])
    # Integrate to velocity
    vel, _ = getvel_dsp(acc_copy2, dt)

    # Compute best fit trend in velocity
    vel_fit1_coef = np.polyfit(t, vel, 1)
    vel_fit2_coef = np.polyfit(t, vel, 2)
    vel_fit1 = vel_fit1_coef[0] * t + vel_fit1_coef[1]
    vel_fit2 = vel_fit2_coef[0] * t * t + vel_fit2_coef[1] * t + vel_fit2_coef[2]
    RMSD1 = np.sqrt(np.mean((vel_fit1 - vel) ** 2))
    RMSD2 = np.sqrt(np.mean((vel_fit2 - vel) ** 2))

    # Remove derivative of best fit trend from accelerationf
    if RMSD1 > RMSD2:
        acc_copy2 = acc_copy2 - (2 * vel_fit2_coef[0] * t + vel_fit2_coef[1])
    else:
        acc_copy2 = acc_copy2 - vel_fit1_coef[0]
        
    # Integrate acceleration to velocity
    vel, _ = getvel_dsp(acc_copy2, dt)

    # Quality check for velocity
    flc = 0.1
    fhc = np.min([40, 0.5 / dt - 5])
    win_len = np.max([loc - loc_uncer, 1 / flc])
    lead = np.abs(np.mean(vel[:int(win_len / dt)]))
    trail = np.abs(np.mean(vel[-int(win_len / dt):]))
    if lead > 0.01 or trail > 0.01:
        print('Quality check for velocity not pass!')

    # Tapering and padding
    N_begin = int((loc - loc_uncer) / dt / 2)
    N_end = int(3 / dt)
    taper_begin = 0.5 * (1 - np.cos(np.pi * np.linspace(0, N_begin - 1, N_begin) / N_begin))
    taper_end = 0.5 * (1 + np.cos(np.pi * np.linspace(0, N_end - 1, N_end) / N_end))
    acc_copy2[:N_begin] = acc_copy2[:N_begin] * taper_begin
    acc_copy2[-N_end:] = acc_copy2[-N_end:] * taper_end

    num_pad = int(6 / flc / dt)
    acc_copy2 = np.concatenate([np.zeros(int(num_pad / 2)), acc_copy2, np.zeros(int(num_pad / 2))])

    # Acausal bandpass filter acceleration
    sos = scipy.signal.butter(4, [flc, fhc], 'bandpass', fs=int(1 / dt), output='sos')
    acc_copy2 = scipy.signal.sosfilt(sos, acc_copy2)

    # Integrate acceleration to velocity and displacement
    vel, dsp = getvel_dsp(acc_copy2, dt)
    acc_copy2 = acc_copy2[int(num_pad / 2) + 2 : len(acc_copy2) - int(num_pad / 2) + 2]
    vel = vel[int(num_pad / 2) + 2 : len(vel) - int(num_pad / 2) + 2]
    dsp = dsp[int(num_pad / 2) + 2 : len(dsp) - int(num_pad / 2) + 2]

    # Quality check for final velocity and displacement
    win_len = np.max([loc - loc_uncer, 1 / flc])
    vel_lead = np.abs(np.mean(vel[:int(win_len / dt)]))
    vel_trail = np.abs(np.mean(vel[-int(win_len / dt):]))
    dsp_trail = np.abs(np.mean(dsp[-int(win_len / dt):]))
    if vel_lead > 0.01 or vel_trail > 0.01 or dsp_trail > 0.01:
        print('Quality check for velocity and displacement not pass!')

    return acc_copy2, vel, dsp