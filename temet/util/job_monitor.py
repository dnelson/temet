"""
Helpers for working with SLURM jobs.
"""

import os
import re


def checkVisJobs():
    """Categorize a large job set into running/completed/failed and automatically re-submit jobs which have failed."""
    startJob = 0
    endJob = 3976  # 3976
    username = "dnelson"

    slurmJobPath = os.path.expanduser("~") + "/ArepoVTK/run.subbox0/"
    slurmJobFile = "job_8k.slurm"

    jobOutputPath = slurmJobPath + "output/frames_8192/"
    outFileRegex = "frame_(.*?)_16bit.png"  #'frame_1820_(.*?)_' + str(nJobs) + '.hdf5'

    tempJobFile = slurmJobPath + "job_temp22.slurm"
    if os.path.isfile(tempJobFile):
        print("Error: Temporary job file exists.")
        return

    # job index lists
    jobsCompleted = []
    jobsRunning = []
    jobsMissing = {str(x) for x in range(startJob, endJob + 1, 1)}

    # load job file and get job naming syntax
    jobFileText = open(slurmJobPath + slurmJobFile).read()

    jobName = re.search(r"^#SBATCH -J (.*?)$", jobFileText, re.M).group(1)

    # get listing of existing hdf5 files (completed jobs)
    files = os.listdir(jobOutputPath)

    for file in files:
        res = re.search(outFileRegex, file)
        if res:
            jobsCompleted.append(res.group(1).lstrip("0"))

    # query slurm for list of running jobs
    slurmText = os.popen(f'squeue -h --array -u {username} -o "%j %K %T"').read()
    slurmText = slurmText.split("\n")

    for line in slurmText:
        jobInfo = line.split(" ")
        if jobInfo[0] == jobName:
            jobsRunning.append(jobInfo[1])

    # any job not running and not finished, add to array string
    jobsMissing -= set(jobsCompleted)
    jobsMissing -= set(jobsRunning)

    arrayLineText = "#SBATCH --array=" + ",".join(sorted(jobsMissing))

    # make new jobfile and launch this new job
    jobFileText = re.sub(r"^#SBATCH --array(.*?)$", arrayLineText, jobFileText, count=1, flags=re.M)

    file = open(tempJobFile, "w")
    file.write(jobFileText)
    file.close()

    print(jobsMissing)
    # execRet = os.popen('sbatch '+tempJobFile).read()
    # print("SBATCH [" + str(len(jobsMissing)) + " new jobs]: " + execRet)

    os.remove(tempJobFile)


def submitExpandedJobs(jobNum):
    """Jobs to build image pyramid for explorer."""
    from ..vis.arepovtk import expandedJobNums

    # config
    totNumJobs = 256
    expansionFac = 16
    slurmJobFile = "job.slurm"  # "job_centos5.bsub"
    tempJobFile = "job22.slurm"
    jobCommand = "sbatch"  # "bsub < "

    # load job template
    jobFileText = open(slurmJobFile).read()

    # get job numbers
    jobNums = expandedJobNums(jobNum, totNumJobs, expansionFac)

    for curJob in jobNums:
        print(curJob)
        # make copy of job script, replace with jobNum
        jobText_local = re.sub(r"NNNN", str(curJob), jobFileText)
        jobText_local = re.sub(r"JJJJ", str(jobNum), jobText_local)

        file = open(tempJobFile, "w")
        file.write(jobText_local)
        file.close()

        # submit and delete
        os.popen(jobCommand + " " + tempJobFile).read()
        os.remove(tempJobFile)
