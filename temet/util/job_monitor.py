"""Helpers for working with SLURM jobs."""

import os
import re
import subprocess
import tempfile


def checkVisJobs():
    """Categorize a large job set into running/completed/failed and automatically re-submit jobs which have failed."""
    startJob = 0
    endJob = 3976  # 3976
    username = "dnelson"

    slurmJobPath = os.path.expanduser("~") + "/ArepoVTK/run.subbox0/"
    slurmJobFile = "job_8k.slurm"

    jobOutputPath = slurmJobPath + "output/frames_8192/"
    outFileRegex = "frame_(.*?)_16bit.png"  #'frame_1820_(.*?)_' + str(nJobs) + '.hdf5'

    # job index lists
    jobsCompleted = []
    jobsRunning = []
    jobsMissing = {str(x) for x in range(startJob, endJob + 1, 1)}

    # load job file and get job naming syntax
    if not os.path.isfile(slurmJobPath + slurmJobFile):
        print(f"Error: Job file [{slurmJobPath + slurmJobFile}] not found.")
        return

    jobFileText = open(slurmJobPath + slurmJobFile).read()
    jobMatch = re.search(r"^#SBATCH -J (.*?)$", jobFileText, re.M)
    if not jobMatch:
        print("Error: Could not find job name in template.")
        return
    jobName = jobMatch.group(1)

    # get listing of existing hdf5 files (completed jobs)
    if os.path.isdir(jobOutputPath):
        files = os.listdir(jobOutputPath)
        for file in files:
            res = re.search(outFileRegex, file)
            if res:
                jobsCompleted.append(res.group(1).lstrip("0"))

    # query slurm for list of running jobs safely
    try:
        slurmText = subprocess.check_output(
            ["squeue", "-h", "--array", "-u", username, "-o", "%j %K %T"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).split("\n")
    except (subprocess.CalledProcessError, FileNotFoundError):
        slurmText = []

    for line in slurmText:
        jobInfo = line.split(" ")
        if len(jobInfo) > 1 and jobInfo[0] == jobName:
            jobsRunning.append(jobInfo[1])

    # any job not running and not finished, add to array string
    jobsMissing -= set(jobsCompleted)
    jobsMissing -= set(jobsRunning)

    arrayLineText = "#SBATCH --array=" + ",".join(sorted(jobsMissing))

    # make new jobfile and launch this new job securely
    jobFileText = re.sub(r"^#SBATCH --array(.*?)$", arrayLineText, jobFileText, count=1, flags=re.M)

    tempJobFile = None
    try:
        target_dir = slurmJobPath if os.path.isdir(slurmJobPath) else None
        with tempfile.NamedTemporaryFile("w", dir=target_dir, prefix="job_temp_", suffix=".slurm", delete=False) as f:
            f.write(jobFileText)
            tempJobFile = f.name

        print(jobsMissing)
        # execRet = subprocess.check_output(["sbatch", tempJobFile], text=True)
        # print("SBATCH [" + str(len(jobsMissing)) + " new jobs]: " + execRet)
    finally:
        if tempJobFile and os.path.isfile(tempJobFile):
            os.remove(tempJobFile)


def submitExpandedJobs(jobNum):
    """Jobs to build image pyramid for explorer."""
    from ..vis.arepovtk import expandedJobNums

    # config
    totNumJobs = 256
    expansionFac = 16
    slurmJobFile = "job.slurm"  # "job_centos5.bsub"
    jobCommand = "sbatch"  # "bsub"

    if jobCommand not in ["sbatch", "bsub"]:
        raise ValueError(f"Unsupported job submission command: {jobCommand}")

    if not os.path.isfile(slurmJobFile):
        print(f"Error: Job template [{slurmJobFile}] not found.")
        return

    # load job template
    jobFileText = open(slurmJobFile).read()

    # get job numbers
    jobNums = expandedJobNums(jobNum, totNumJobs, expansionFac)

    for curJob in jobNums:
        print(curJob)
        # make copy of job script, replace with jobNum
        jobText_local = re.sub(r"NNNN", str(curJob), jobFileText)
        jobText_local = re.sub(r"JJJJ", str(jobNum), jobText_local)

        tempJobFile = None
        try:
            with tempfile.NamedTemporaryFile("w", prefix="job_", suffix=".slurm", delete=False) as f:
                f.write(jobText_local)
                tempJobFile = f.name

            # submit and delete
            subprocess.run([jobCommand, tempJobFile], check=True, text=True)
        finally:
            if tempJobFile and os.path.isfile(tempJobFile):
                os.remove(tempJobFile)
