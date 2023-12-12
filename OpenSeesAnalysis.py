import subprocess
import os
import numpy as np


def getProfile(filepath):
    f = open(filepath)
    data = f.readlines()
    f.close()
    data = data[2:]
    profile = np.zeros((len(data), 3))
    for i, line in enumerate(data):
        line = line.replace('\n', '').replace(',', ' ').split()
        profile[i, 1] = float(line[3])
        profile[i, 2] = float(line[4])
        if i < len(data) - 1:
            profile[i, 0] = float(line[2])
        else:
            profile[i, 0] = -1
    return profile


def change_analysis_time(tclfile, numstep, dt):
    '''
    功能：此函数用于修改analysis.tcl（OpenSees输入文件中分析部分）中的分析时间步数

    --Input
    -tclfile: string, tcl文件路径
    -numstep: int, 分析时间步数
    -dt: float, 分析时间步长

    --Return
    无
    '''
    analysisfile = open(tclfile, 'r')
    content = analysisfile.read()
    analysisfile.close()
    content = content.split('\n')
    analysisfile = open(tclfile, 'w')
    for line in content:
        if 'set numstep' in line:
            analysisfile.write('set numstep %d\n' % numstep)
        elif 'set dt' in line:
            analysisfile.write('set dt %.3f\n' % dt)
        else:
            analysisfile.write(line)
            analysisfile.write('\n')
    analysisfile.close()


def change_analysis(exam_path, tgt_path, inp, out, numstep, dt):
    exam_file = open(exam_path)
    content = exam_file.read()
    exam_file.close()
    content = content.replace('inp_path', inp)
    content = content.replace('out_path', out)
    content = content.replace('num_step', '%d' % numstep)
    content = content.replace('--dt', '%f' % dt)
    tgt_file = open(os.path.join(tgt_path, 'analysis.tcl'), 'w')
    tgt_file.write(content)
    tgt_file.close()


def change_paras(exam_path, tgt_path, vsdata):
    numLayer = int(vsdata.shape[0])
    rockLayer = int(sum(vsdata[:, 1] >= 1000))
    softLayer = int(sum(vsdata[:, 1] < 400))
    f = open(exam_path)
    content = f.read()
    f.close()
    content = content.split('\n')
    tgt_file = open(tgt_path, 'w')
    for line in content:
        if 'set Vs(i) data' in line:
            for i in range(numLayer):
                tgt_file.write('set Vs(%d) %.1f\n' % (numLayer - i, vsdata[i, 1]))
        elif '--numLayers' in line:
            tgt_file.write('set numLayers %d\n' % numLayer)
        elif 'set layerThick(i) h' in line:
            tgt_file.write('set layerThick(%d) %.1f\n' % (numLayer, np.round(vsdata[0, 0])))
            for i in range(1, numLayer):
                tgt_file.write('set layerThick(%d) %.1f\n' % (numLayer - i, np.round(vsdata[i, 0] - vsdata[i - 1, 0])))
        elif '--RockLayers' in line:
            tgt_file.write('set RockLayers %d\n' % rockLayer)
        elif '--SoftLayers' in line:
            tgt_file.write('set SoftLayers %d\n' % softLayer)
        else:
            tgt_file.write('%s\n' % line)
    tgt_file.close()


def runOpenSees(filepath):
    '''
    功能：此函数用于控制OpenSees进行场地响应计算

    --Input
    -tclfile:string, 输入tcl文件名

    --Return
    无
    '''
    command = 'source main.tcl'
    # s = subprocess.Popen(os.path.join(filepath, 'OpenSees.exe'), shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    s = subprocess.Popen('OpenSees.exe', shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=filepath)
    s.stdin.write(command.encode())
    s.communicate()
    s.terminate()