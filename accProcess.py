import shutil
import tarfile
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from tqdm import tqdm
import matlab.engine
matlabeng = matlab.engine.start_matlab()   #启动matlab


def getIntDifMat(n, dt):
    phi1 = np.concatenate([np.array([0, 0.5, 0]), np.zeros([n - 3, ])])
    temp1 = np.concatenate([-1 / 2 * np.identity(n - 2), np.zeros([n - 2, 2])], axis=1)
    temp2 = np.concatenate([np.zeros([n - 2, 2]), 1 / 2 * np.identity(n - 2)], axis=1)
    phi2 = temp1 + temp2
    phi3 = np.concatenate([np.zeros([n - 3, ]), np.array([0, -0.5, 0])])
    Phi_dif = np.concatenate([np.reshape(phi1, [1, phi1.shape[0]]), phi2, np.reshape(phi3, [1, phi3.shape[0]])], axis=0)
    Phi_int = np.linalg.inv(Phi_dif) * dt
    Phi_dif = Phi_dif / dt
    return Phi_int, Phi_dif


def getIntDifMat1(n, dt):
    Phi_int = np.triu(np.zeros((n, n)) + 1).transpose() - np.identity(n) / 2
    Phi_dif = np.linalg.inv(Phi_int) / dt
    Phi_int *= dt
    return Phi_int, Phi_dif


def getacc(filename):
    file = open(filename, 'r')
    acc = file.read()
    file.close()
    acc = acc.split()
    acc = [float(a) for a in acc]
    acc = np.array(acc).ravel()
    return acc


def getvel_dsp(acc, dt):
    n = acc.shape[-1]
    if n < 10000:
        Phi_int = np.triu(np.zeros((n, n)) + 1).transpose() - np.identity(n) / 2
        # Phi_dif = np.linalg.inv(Phi_int) / dt
        Phi_int *= dt

        vel = Phi_int.dot(acc.transpose()).transpose()
        if len(vel.shape) == 2:
            vel = vel - np.mean(vel, axis=1)[:, None]
        else:
            vel = vel - np.mean(vel)
        dsp = Phi_int.dot(vel.transpose()).transpose()
        if len(dsp.shape) == 2:
            dsp = dsp - np.mean(dsp, axis=1)[:, None]
        else:
            dsp = dsp - np.mean(dsp)
    else:
        vel = np.zeros_like(acc)
        vel[..., 0] = 0.5 * acc[..., 0] * dt
        for i in range(1, n):
            vel[..., i] = vel[..., i - 1] + 0.5 * (acc[..., i - 1] + acc[..., i]) * dt
        vel = vel - np.mean(vel)
        dsp = np.zeros_like(vel)
        dsp[..., 0] = 0.5 * vel[..., 0] * dt
        for i in range(1, n):
            dsp[..., i] = dsp[..., i - 1] + 0.5 * (vel[..., i - 1] + vel[..., i]) * dt
        dsp = dsp - np.mean(dsp)
    return vel, dsp


def baselineCorrection(acc, dt, M):
    t = np.linspace(dt, dt * len(acc), len(acc))
    acc1 = acc
    vel, dsp = getvel_dsp(acc, dt)
    Gv = np.zeros(shape=(acc.shape[0], M + 1))
    for i in range(M + 1):
        Gv[:, i] = t ** (M + 1 - i)
    polyv = np.dot(np.dot(np.linalg.inv(Gv.transpose().dot(Gv)), Gv.transpose()), vel)
    for i in range(M + 1):
        acc1 -= (M + 1 - i) * polyv[i] * t ** (M - i)
        
    acc_new = acc1
    vel1, dsp1 = getvel_dsp(acc1, dt)
    Gd = np.zeros(shape=(acc.shape[0], M + 1))
    for i in range(M + 1):
        Gd[:, i] = t ** (M + 2 - i)
    polyd = np.dot(np.dot(np.linalg.inv(Gd.transpose().dot(Gd)), Gd.transpose()), dsp1)
    for i in range(M + 1):
        acc_new -= (M + 2 - i) * (M + 1 - i) * polyd[i] * t ** (M - i)
    return acc_new


def solve_sdof_eqwave_piecewise_exact(omg, zeta, ag, dt):
    omg_d = omg * np.sqrt(1.0 - zeta * zeta)
    m = len(omg)
    n = len(ag)
    u, v = np.zeros((m, n)), np.zeros((m, n))
    B1 = np.exp(-zeta * omg * dt) * np.cos(omg_d * dt)
    B2 = np.exp(-zeta * omg * dt) * np.sin(omg_d * dt)
    omg2 = 1.0 / omg / omg
    omg3 = 1.0 / omg / omg / omg
    for i in range(n - 1):
        alpha = (-ag[i + 1] + ag[i]) / dt
        A0 = -ag[i] * omg2 - 2.0 * zeta * alpha * omg3
        A1 = alpha * omg2
        A2 = u[:, i] - A0
        A3 = (v[:, i] + zeta * omg * A2 - A1) / omg_d
        u[:, i + 1] = A0 + A1 * dt + A2 * B1 + A3 * B2
        v[:, i + 1] = A1 + (omg_d * A3 - zeta * omg * A2) * B1 - (omg_d * A2 + zeta * omg * A3) * B2
    return u, v


def getResponseSpectrum(acc, dt, Period=np.logspace(-1.5, 0.5, 300), damp=0.05):
    sa = matlabeng.getResponseSpectrum(matlab.double(acc.tolist()), dt, matlab.double(Period.tolist()), damp)
    return np.array(sa).ravel()


def response_spectra_py(ag, dt, T=np.logspace(-1.5, 0.5, 300), zeta=0.05):
    N = len(T)
    RSA = np.zeros(N)
    RSV = np.zeros(N)
    RSD = np.zeros(N)
    omg = 2.0 * np.pi / T
    u, v = solve_sdof_eqwave_piecewise_exact(omg, zeta, ag, dt)
    a = -2.0 * zeta * omg[:, None] * v - omg[:, None]  * omg[:, None]  * u
    RSA = np.max(np.abs(a), axis=1)
    RSV = np.max(np.abs(v), axis=1)
    RSD = np.max(np.abs(u), axis=1)
    return RSA, RSV, RSD


def getFourierSpectrum(acc, dt, freq=None, smooth=None, coef=None):
    num = len(acc)
    newdt = 0.01
    newnum = np.floor(num * dt / newdt)
    acc = np.interp((np.arange(newnum) + 1) * newdt, (np.arange(num) + 1) * dt, acc)
    nfft = int(2 ** np.ceil(np.log2(newnum)))
    h = np.fft.fft(acc, n=nfft)
    f = np.fft.fftfreq(nfft, d=newdt)
    h = np.abs(h[1 : int(len(h) / 2)])
    f = f[1 : int(len(f) / 2)]
    if smooth == 'kohmachi':
        h = kohmachi(h, f, coef)
    if smooth == 'movemean':
        h = np.convolve(h, np.ones((coef,)) / coef, mode='same')
    if freq is not None:
        h = np.interp(freq, f, h)
        f = freq
    return f, h


def kohmachi(sign, freq, coef):
    f_shift = freq / (1 + 1e-4)
    f_shift = np.repeat(f_shift[:, None], len(f_shift), axis=1)
    z = f_shift / freq
    w = (np.sin(coef * np.log10(z)) / (coef * np.log10(z))) ** 4
    y = sign.dot(w) / np.sum(w, axis=0)
    return y


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


def cvt2col(filename):
    '''
    将KiK-net的原始数据转换为列数据
    '''
    f = open(filename, 'r')
    for i in range(10):
        line = f.readline()
    
    line = f.readline()
    freq = line.split()[-1]
    freq = int(freq[:-2])
    dt = 1 / freq
    for i in range(2):
        line = f.readline()

    line = f.readline()
    line = line.split()[-1]
    line = line.split('/')
    scft = float(line[0][:-5]) / float(line[1])
    line = f.readline()
    maxa = float(line.split()[-1])
    for i in range(2):
        line = f.readline()

    acc = f.read()
    acc = acc.split()
    acc = [scft * float(a) for a in acc]
    acc = np.array(acc)
    acc = acc - np.mean(acc)
    acc = acc / 981
    f.close()

    if filename.endswith('.EW1'):
        id1 = 'EW_dh'
    elif filename.endswith('.EW2'):
        id1 = 'EW_up'
    elif filename.endswith('.NS1'):
        id1 = 'NS_dh'
    elif filename.endswith('.NS2'):
        id1 = 'NS_up'
    elif filename.endswith('.UD1'):
        id1 = 'UD_dh'
    elif filename.endswith('.UD2'):
        id1 = 'UD_up'
    else:
        id1 = 'other'
    
    if abs(dt - 0.01) < 1e-5:
        id2 = '_010.acc'
    elif abs(dt - 0.005) < 1e-5:
        id2 = '_005.acc'
    else:
        id2 = '_dt.acc'
    
    cvtfilename = filename[:-4] + id1 + id2
    f = open(cvtfilename, 'w')
    for a in acc:
        f.write('{:7.6E}\n'.format(a))

    f.close()
    return acc, dt


def getazimuth(station, source):
    '''
    station: 测站的经纬度[纬度, 经度]
    source: 震源的经纬度[纬度, 经度]
    '''
    direc = np.zeros(2)
    direc[0] = source[0] - station[0]
    direc[1] = (source[1] - station[1]) * np.cos(source[0] * np.pi / 180)
    direc = direc / np.linalg.norm(direc)
    return direc

def getSH(filename, station, source, dt):
    direc = getazimuth(station, source)
    if np.abs(dt - 0.005) < 1e-5:
        dh_EW = getacc(filename + 'EW_dh_005.acc')
        dh_NS = getacc(filename + 'NS_dh_005.acc')
        dh_file = open(filename + '_dh_005.acc', 'w')
        up_EW = getacc(filename + 'EW_up_005.acc')
        up_NS = getacc(filename + 'NS_up_005.acc')
        up_file = open(filename + '_up_005.acc', 'w')
    elif np.abs(dt - 0.01) < 1e-5:
        dh_EW = getacc(filename + 'EW_dh_010.acc')
        dh_NS = getacc(filename + 'NS_dh_010.acc')
        dh_file = open(filename + '_dh_010.acc', 'w')
        up_EW = getacc(filename + 'EW_up_010.acc')
        up_NS = getacc(filename + 'NS_up_010.acc')
        up_file = open(filename + '_up_010.acc', 'w')
    else:
        dh_EW = getacc(filename + 'EW_dh_dt.acc')
        dh_NS = getacc(filename + 'NS_dh_dt.acc')
        dh_file = open(filename + '_dh_dt.acc', 'w')
        up_EW = getacc(filename + 'EW_up_dt.acc')
        up_NS = getacc(filename + 'NS_up_dt.acc')
        up_file = open(filename + '_up_dt.acc', 'w')

    dh_acc = np.zeros_like(dh_EW)
    for i in range(len(dh_acc)):
        dh_acc[i] = dh_EW[i] * direc[0] - dh_NS[i] * direc[1]

    for a in dh_acc:
        dh_file.write('{:7.6E}\n'.format(a))
    dh_file.close()

    up_acc = np.zeros_like(up_EW)
    for i in range(len(dh_acc)):
        up_acc[i] = up_EW[i] * direc[0] - up_NS[i] * direc[1]

    for a in up_acc:
        up_file.write('{:7.6E}\n'.format(a))
    up_file.close()


# # 解压文件
# flist = os.listdir()
# for file in flist:
#     if file.endswith('.tar.gz'):
#         name = file.split('.')[0]
#         t = tarfile.open(file)
#         t.extractall(name)
#         t.close()

# # 将KiK-net格式的文件转换为列
# label = ['.EW1', '.EW2', '.NS1', '.NS2', '.UD1', '.UD2']
# flist = os.listdir()
# pbar = tqdm(flist, desc='转换Kik-net格式文件', ncols=100)
# for filedir in pbar:
#     if os.path.isdir(filedir):
#         for f in os.listdir(filedir):
#             if f[-4:] in label:
#                 cvt2col(os.path.join(filedir, f))

# # 旋转EW和NS分量
# print()
# filelist = os.listdir()
# pbar = tqdm(filelist, desc='旋转EW和NS分量', ncols=100)
# for filedir in pbar:
#     if os.path.isdir(filedir) and filedir != '.vscode':
#         EWfile = os.path.join(filedir, filedir + '.EW1')
#         sor = []
#         sta = []
#         f = open(EWfile, 'r')
#         for line in f.readlines():
#             if 'Lat' in line and 'Station' not in line:
#                 sor.append(float(line.split()[-1]))
#             if 'Long' in line and 'Station' not in line:
#                 sor.append(float(line.split()[-1]))
#             if 'Station Lat' in line:
#                 sta.append(float(line.split()[-1]))
#             if 'Station Long' in line:
#                 sta.append(float(line.split()[-1]))
#             if 'Sampling Freq' in line:
#                 freq = line.split()[-1]
#                 dt = 1 / float(freq[:-2])
#                 break
#         f.close()
#         shfile = os.path.join(filedir, filedir)
#         getSH(shfile, sta, sor, dt)

# # 绘图
# print()
# filelist = os.listdir()
# pbar = tqdm(filelist, desc='绘图', ncols=100)
# for filedir in pbar:
#     if os.path.isdir(filedir) and filedir != '.vscode':
#         flist = os.listdir(filedir)
#         for f in flist:
#             if f.endswith('.acc') and 'filter' not in f:
#                 name = f.split('.')[0]
#                 name = name.split('_')
#                 if name[-1] == '005':
#                     dt = 0.005
#                 elif name[-1] == '010':
#                     dt = 0.01
#                 else:
#                     dt = 10
#                 if name[0][-2:] == 'EW':
#                     if name[1] == 'dh':
#                         EW_dh = getacc(os.path.join(filedir, f))
#                     else:
#                         EW_up = getacc(os.path.join(filedir, f))
#                 elif name[0][-2:] == 'NS':
#                     if name[1] == 'dh':
#                         NS_dh = getacc(os.path.join(filedir, f))
#                     else:
#                         NS_up = getacc(os.path.join(filedir, f))
#                 elif name[0][-2:] == 'UD':
#                     if name[1] == 'dh':
#                         UD_dh = getacc(os.path.join(filedir, f))
#                     else:
#                         UD_up = getacc(os.path.join(filedir, f))
#                 else:
#                     if name[1] == 'dh':
#                         dh = getacc(os.path.join(filedir, f))
#                     else:
#                         up = getacc(os.path.join(filedir, f))

#         T = np.linspace(dt, dt * len(EW_dh), len(EW_dh))
#         plt.plot(T, EW_up, T, EW_dh, linewidth=0.5)
#         plt.xlabel('t(s)')
#         plt.ylabel('acc(g)')
#         plt.legend(['surface', 'downhole'])
#         plt.savefig(os.path.join(filedir, filedir + 'EW.png'), dpi=300)
#         plt.close()

#         plt.plot(T, NS_up, T, NS_dh, linewidth=0.5)
#         plt.xlabel('t(s)')
#         plt.ylabel('acc(g)')
#         plt.legend(['surface', 'downhole'])
#         plt.savefig(os.path.join(filedir, filedir + 'NS.png'), dpi=300)
#         plt.close()

#         plt.plot(T, UD_up, T, UD_dh, linewidth=0.5)
#         plt.xlabel('t(s)')
#         plt.ylabel('acc(g)')
#         plt.legend(['surface', 'downhole'])
#         plt.savefig(os.path.join(filedir, filedir + 'UD.png'), dpi=300)
#         plt.close()

#         plt.plot(T, up, T, dh, linewidth=0.5)
#         plt.xlabel('t(s)')
#         plt.ylabel('acc(g)')
#         plt.legend(['surface', 'downhole'])
#         plt.savefig(os.path.join(filedir, filedir + '.png'), dpi=300)
#         plt.close()