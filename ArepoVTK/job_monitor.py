# job_monitor.py
# dnelson
# jan.2014

# 1820_128k_sphTree (31 jan): SBATCH [186 new jobs]: Submitted batch job 5722986
#                   (10 feb): SBATCH [144 new jobs]: Submitted batch job 6266169

import re
import os

def checkVisJobs():
    nJobs = 256
    slurmJobPath = '/n/home07/dnelson/ArepoVTK/run.illustris.box/'
    slurmJobFile = 'job_1820_256_128k_sphTree.slurm'
    
    jobOutputPath = slurmJobPath + 'output/1820_256_128k_sphTree/'
    outFileRegex = 'frame_1820_(.*?)_' + str(nJobs) + '.hdf5'
    
    tempJobFile = slurmJobPath + 'job_temp22.slurm'
    if os.path.isfile(tempJobFile):
        print("Error: Temporary job file exists.")
        return
    
    # job index lists
    jobsCompleted = []
    jobsRunning = []
    jobsMissing = set( [str(x) for x in range(nJobs)] )
    
    # load job file and get job naming syntax
    jobFileText = open(slurmJobPath + slurmJobFile, 'r').read()
    
    jobName = re.search(r'^#SBATCH -J (.*?)$',jobFileText,re.M).group(1)
    
    # get listing of existing hdf5 files (completed jobs)
    files = os.listdir(jobOutputPath)
    
    for i,file in enumerate(files):
        res = re.search(outFileRegex, file)
        if res:
            jobsCompleted.append( res.group(1) )
    
    # query slurm for list of running jobs
    slurmText = os.popen('squeue -h --array -u dnelson -o "%j %K %T"').read()
    slurmText = slurmText.split('\n')
    
    for i,jobText in enumerate(slurmText):
        jobInfo = jobText.split(' ')
        if jobInfo[0] == jobName:
            jobsRunning.append( jobInfo[1] )
    
    # any job not running and not finished, add to array string
    jobsMissing -= set(jobsCompleted)
    jobsMissing -= set(jobsRunning)
    
    arrayLineText = "#SBATCH --array=" + ','.join( sorted(jobsMissing) )
    
    # make new jobfile and launch this new job
    jobFileText = re.sub(r'^#SBATCH --array(.*?)$',arrayLineText,jobFileText,1,re.M)
    
    file = open(tempJobFile, 'w')
    file.write(jobFileText)
    file.close()
    
    execRet = os.popen('sbatch '+tempJobFile).read()
    print("SBATCH [" + str(len(jobsMissing)) + " new jobs]: " + execRet)
    
    os.remove(tempJobFile)
    
