"""
Non-science and other meta plots.
"""

from datetime import datetime
from os.path import expanduser

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter, DayLocator
from scipy.signal import savgol_filter

from ..plot.config import figsize, linestyles, lw
from ..plot.util import getWhiteBlackColors, setAxisColors


def plotUsersData():
    """Parse and plot a user data dump from the Illustris[TNG] public data release website."""
    # config
    col_headers = [
        "Date",
        "NumUsers",
        "Num30",
        "NumLab",
        "CountLogin",
        "CountApi",
        "CountFits",
        "CountLabLogins",
        "CountSnapUni",
        "CountSnapSub",
        "SizeUni",
        "SizeSub",
        "CountGroup",
        "CountLHaloTree",
        "CountSublink",
        "CutoutSubhalo",
        "CutoutSublink",
    ]
    labels = [
        "Total Users",
        "Users Active in Last 30 Days",
        "Users with TNG-Lab Access",
        "Total Logins",
        "Total API Requests",
        "FITS File Downloads",  # / $10^3$
        "Total Lab Logins",
        "# Snapshots Downloaded",  #  / $10^2$
        "# Snapshots [Subbox]",
        "Download Volume [TB]",
        "Total Download Size: Subbox [TB]",
        "# Catalogs Downloaded",
        "Number of Downloads: LHaloTree",
        "Number of Downloads: Sublink",
        "Cutout Requests: Subhalos",
        "Cutout Requests: Sublink",
    ]

    pStyle = "white"  # 'white', 'black'
    fac2 = 1.05

    # load
    def convertfunc(x):
        return datetime.strptime(x, "%Y-%m-%d")

    # dd = [(col_headers[0], 'object')] + [(a, 'd') for a in col_headers[1:]]
    dd = [object] + ["d" for a in col_headers[1:]]
    data_load = np.genfromtxt(
        expanduser("~") + "/plot_stats.txt",
        delimiter=",",
        names=col_headers,
        dtype=dd,
        converters={"Date": convertfunc},
        skip_header=50,
    )

    data = {}
    w = np.where(data_load["NumUsers"] >= 275)  # clip pre-release activity to zero for visuals
    for key in col_headers:
        data[key] = data_load[key][w]
    data["NumUsers"][0] = 1

    # colortable
    tableau20 = [
        (31, 119, 180),
        (174, 199, 232),
        (255, 127, 14),
        (255, 187, 120),
        (44, 160, 44),
        (152, 223, 138),
        (214, 39, 40),
        (255, 152, 150),
        (148, 103, 189),
        (197, 176, 213),
        (140, 86, 75),
        (196, 156, 148),
        (227, 119, 194),
        (247, 182, 210),
        (127, 127, 127),
        (199, 199, 199),
        (188, 189, 34),
        (219, 219, 141),
        (23, 190, 207),
        (158, 218, 229),
    ]
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255.0, g / 255.0, b / 255.0)

    color1, color2, _, _ = getWhiteBlackColors(pStyle)

    # plot (1) - everything
    fig = plt.figure(figsize=(22, 13), facecolor=color1)
    ax = fig.add_subplot(111, facecolor=color1)
    ax.set_yscale("log")

    setAxisColors(ax, color2)

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # ax.tick_params(axis='both', which='major', labelsize=14)
    # ax.set_axis_bgcolor( (1.0,1.0,1.0) )
    ax.set_ylim([1, 1e10])

    launch_date = datetime.strptime("2015-04-01", "%Y-%m-%d")
    launch_date_tng = datetime.strptime("2018-12-17", "%Y-%m-%d")
    ax.plot([launch_date, launch_date], [2, 1e4], "-", lw=lw, color=color2, alpha=0.6)
    ax.plot([launch_date_tng, launch_date_tng], [2, 1e4], "-", lw=lw, color=color2, alpha=0.6)

    for i in range(len(col_headers) - 1):
        col = col_headers[i + 1]
        label = labels[i]

        fac = 1e3 if "[TB]" in label else 1.0
        ls = linestyles[i % len(linestyles)]

        ax.plot_date(data["Date"], data[col] / fac, marker=None, linestyle=ls, label=label, lw=lw, color=tableau20[i])

        if col != "SizeUni" and col != "SizeSub":
            ax.text(
                data["Date"][-1],
                data[col][-1] / fac * fac2,
                str(int(data[col][-1])),
                horizontalalignment="right",
                color=tableau20[i],
            )
        else:
            ax.text(
                data["Date"][-1],
                data[col][-1] / fac * fac2,
                f"{data[col][-1] / fac:.1f}",
                horizontalalignment="right",
                color=tableau20[i],
            )

    ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))
    l = ax.legend(loc="best", ncol=3, frameon=False)
    for text in l.get_texts():
        text.set_color(color2)
    fig.autofmt_xdate()

    fig.savefig("userstats_all.pdf", facecolor=fig.get_facecolor())
    plt.close(fig)

    # plot (2) - user counts
    fig = plt.figure(figsize=figsize, facecolor=color1)
    ax = fig.add_subplot(111, facecolor=color1)
    ax.set_yscale("log")
    ax.set_ylabel("Number")
    ax.yaxis.set_ticks_position("both")
    setAxisColors(ax, color2)

    ax.set_ylim([1e1, 1e5])

    launch_date2 = datetime.strptime("2015-05-01", "%Y-%m-%d")
    launch_date_tng2 = datetime.strptime("2019-01-17", "%Y-%m-%d")
    ax.plot([launch_date, launch_date], [10, 13], "-", lw=14, color=color2, alpha=0.3)
    ax.text(launch_date2, 40, "Illustris Public Data Release", color=color2, alpha=0.3, rotation=20)
    ax.plot([launch_date_tng, launch_date_tng], [10, 13], "-", lw=14, color=color2, alpha=0.3)
    ax.text(launch_date_tng2, 30, "TNG Data Release", color=color2, alpha=0.3, rotation=20)

    for col in ["Num30", "NumLab", "CountSnapUni", "CountGroup", "SizeUni", "NumUsers"]:
        w = col_headers.index(col) - 1
        fac = 1e3 if "[TB]" in labels[w] else 1.0
        if "Count" in col:
            fac = 1

        (l,) = ax.plot_date(data["Date"], data[col] / fac, marker=None, linestyle="-", label=labels[w], lw=lw)
        ax.text(
            data["Date"][-1],
            data[col][-1] / fac * 0.8,
            "%d" % (data[col][-1] / fac),
            color=l.get_color(),
            fontsize=20,
            horizontalalignment="center",
        )

    ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))
    # ax.xaxis.set_minor_formatter(DateFormatter('%B'))
    # ax.xaxis.set_minor_locator(MonthLocator(bymonth=6))
    # ax.tick_params(axis='x', which='minor', labelsize=13)
    l = ax.legend(loc="upper left", ncol=1, frameon=False)
    for text in l.get_texts():
        text.set_color(color2)
    fig.autofmt_xdate(which="both")

    fig.savefig("userstats_users.pdf", facecolor=fig.get_facecolor())
    plt.close(fig)


def plotNumPublicationsVsTime():
    """Compare the number of publications vs time for various projects."""
    import json
    from datetime import datetime
    from re import findall
    from time import mktime

    import requests

    num_start = 10  # align 'time=0' after this number of publications have appeared
    xlim = [-0.6, 8.0]  # years, [-3.2, 7.0] for num_start=100

    today_ts = mktime(datetime.now().timetuple())

    pStyle = "white"  # 'white', 'black'
    color1, color2, _, _ = getWhiteBlackColors(pStyle)

    # load Illustris and TNG
    pub_sets = {
        "Illustris": json.loads(requests.get("https://www.illustris-project.org/results/?json=1").content),
        "TNG": json.loads(requests.get("https://www.tng-project.org/results/?json=1").content),
    }

    for pub_set in pub_sets.values():
        for pub in pub_set:
            pub["ts"] = mktime(datetime.strptime(pub["arxiv_date"], "%Y-%m-%d").timetuple())

    # load Millennium
    if 0:
        pubs_html = requests.get("https://wwwmpa.mpa-garching.mpg.de/millennium/").content.decode()
    else:
        with open("page.html") as f:
            pubs_html = f.read()

    pubs_html = pubs_html[pubs_html.find("<small>PUBLICATIONS</small>") :]

    astroph_ids = findall(r"a name=\"astro-ph/[0-9.]+\"", pubs_html)
    astroph_ids = [apid.split("/")[1].replace('"', "") for apid in astroph_ids]

    pub_sets["Millennium"] = []
    for apid in astroph_ids:
        # possible formats (the first used, after 1 April 2007)
        # "YYMM.NNNN[N]" where YY is a 2-digit year, MM is a 2-digit month, and NNNN is a 4 or 5 digit identifier
        # "YYMMNNN" where YY is a 2-digit year, MM is a 2-digit month, and NNN is the paper index in that month
        # regardless, we have only month time resolution unless we load all the arxiv pages and scrape the dates
        pub = {"ts": mktime(datetime.strptime(apid[0:4], "%y%m").timetuple())}
        pub_sets["Millennium"].append(pub)

    # start plot
    fig = plt.figure(figsize=(figsize[0] * 0.8, figsize[1] * 0.8), facecolor=color1)
    ax = fig.add_subplot(111, facecolor=color1)
    ax.set_ylabel("Number of Publications")
    ax.set_xlabel("Number of Years since %s Publication" % ("%d$^{\\rm th}$" % num_start if num_start > 0 else "First"))
    ax.set_axisbelow(True)
    ax.grid(alpha=0.3)
    setAxisColors(ax, color2)

    ax.set_xlim(xlim)
    ax.set_ylim([0, len(pub_sets["TNG"]) * 1.05])
    # ax.set_ylim([10, 1200])
    # ax.set_yscale('log')

    for sim_name, pub_set in pub_sets.items():
        xx = np.array([pub["ts"] for pub in pub_set])
        xx = xx[np.argsort(xx)]  # make sure we are sorted ascending in time

        # calculate starting point, and delta years of each publication
        start_ts = xx[num_start]
        xx_plot = (xx - start_ts) / (60 * 60 * 24 * 7 * 52)  # delta years

        ax.plot(xx_plot, np.arange(xx_plot.size), "-", lw=lw, label=sim_name)
        print(sim_name, len(pub_set))

    l = ax.legend(loc="upper left")
    for text in l.get_texts():
        text.set_color(color2)
    fig.savefig("numpubs_vs_time_%d.pdf" % num_start, facecolor=fig.get_facecolor())
    plt.close(fig)

    # compute number of publications 'per day'
    fig = plt.figure(figsize=(figsize[0] * 0.7, figsize[1] * 0.7), facecolor=color1)
    ax = fig.add_subplot(111, facecolor=color1)
    ax.set_ylabel("Papers per arXiv day")
    ax.set_xlabel("Number of Years since %s Publication" % ("%d$^{\\rm th}$" % num_start if num_start > 0 else "First"))
    ax.set_axisbelow(True)
    ax.grid(alpha=0.3)
    setAxisColors(ax, color2)

    ax.set_xlim(xlim)
    ax.set_ylim([0.0, 1.2])  # len(pub_sets['TNG'])*1.05])

    for sim_name, pub_set in pub_sets.items():
        xx = np.array([pub["ts"] for pub in pub_set])
        xx = xx[np.argsort(xx)]  # make sure we are sorted ascending in time

        # calculate starting point, and delta years of each publication
        start_ts = xx[num_start]
        xx_plot = (xx - start_ts) / (60 * 60 * 24 * 7 * 52)  # delta days

        # in days
        xx_days = (xx - start_ts) / (60 * 60 * 24)
        num_per_day = np.zeros(xx_days.size, dtype="float32")

        for i in range(xx_days.size):
            cur_day = xx_days[i]
            num_weeks_sm = 12  # four week smoothing window
            min_day = cur_day - (num_weeks_sm / 2 * 7)
            max_day = cur_day + (num_weeks_sm / 2 * 7)
            w = np.where((xx_days >= min_day) & (xx_days < max_day))[0]
            num_per_day[i] = len(w) / (num_weeks_sm * 5)  # 'arxiv days' i.e. weekdays

            # if future is incomplete, truncate
            today = (today_ts - start_ts) / (60 * 60 * 24)  # days
            if max_day > today:
                num_per_day[i] = np.nan

        ax.plot(xx_plot, num_per_day, "-", lw=lw, label=sim_name)
        print(sim_name, len(pub_set))

    l = ax.legend(loc="upper left")
    for text in l.get_texts():
        text.set_color(color2)
    fig.savefig("numpubs_perday_%d.pdf" % num_start, facecolor=fig.get_facecolor())
    plt.close(fig)


def plotCpuTimeEstimates():
    """Plot predicted total CPUh and finish date, as a function of prediction date, given the plotCpuTimes() log."""
    fName1 = expanduser("~") + "/plots/cpu_estimated.pdf"
    fName2 = expanduser("~") + "/lsf/crontab/cpu_tng.log"
    runName = "TNG50-1"

    lw = 2.0
    date_fmt = "%d %B, %Y"
    xlim_dates = [datetime.strptime(d, date_fmt) for d in ["01 February, 2017", "01 July, 2017"]]
    ylim_dates = [datetime.strptime(d, date_fmt) for d in ["01 June, 2017", "01 March, 2018"]]
    start_date = datetime.strptime("21 February, 2017", date_fmt)

    dates = []
    cpuhs = []
    finish_dates = []

    # newer additions
    dates_extra = []
    cpuhs_tng100 = []
    finish_dates_tng100 = []
    cpuhs_1024 = []
    finish_dates_1024 = []

    # load and parse
    readNextAsPredict = False
    iterUntilNextDate = True

    f = open(fName2)
    lines = f.readlines()

    for line in lines:
        line = line.strip()

        # daily run header
        if line[0:7] == "-- run:":
            date = line.split("run: ")[1].split(" --")[0].replace("Feb", "February")
            date = datetime.strptime(date, date_fmt)
            dates.append(date)
            iterUntilNextDate = False

        if iterUntilNextDate:
            continue

        if line[0 : len(runName) + 9] == "%s [total]:" % runName:
            readNextAsPredict = True
            continue

        # e.g. "Predicted total time: 104.8 million CPUhs (21 November, 2017)" after "TNG50-1 [total]:*" line
        if readNextAsPredict:
            if "Predicted total time:" in line:
                cpuh = float(line.split("time: ")[1].split(" ")[0])
                finish_date = line.split("(")[1].split(")")[0]
                finish_date = datetime.strptime(finish_date, date_fmt)
                cpuhs.append(cpuh)
                finish_dates.append(finish_date)
            elif "[w/ TNG100-1] Predicted:" in line:
                cpuh = float(line.split("Predicted: ")[1].split(" ")[0])
                finish_date = line.split("(")[1].split(")")[0]
                finish_date = datetime.strptime(finish_date, date_fmt)
                cpuhs_tng100.append(cpuh)
                finish_dates_tng100.append(finish_date)
                dates_extra.append(dates[-1])
            elif "[w/ L25n1024_4503] Predicted:" in line:
                cpuh = float(line.split("Predicted: ")[1].split(" ")[0])
                finish_date = line.split("(")[1].split(")")[0]
                finish_date = datetime.strptime(finish_date, date_fmt)
                cpuhs_1024.append(cpuh)
                finish_dates_1024.append(finish_date)
            else:
                readNextAsPredict = False
                iterUntilNextDate = True

    f.close()

    # start plot
    fig = plt.figure(figsize=(16, 14))
    # (A)
    ax = fig.add_subplot(211)

    ax.set_xlim(xlim_dates)
    ax.set_ylim([50, 210])
    ax.set_xlabel("Date")
    ax.set_ylabel("Estimated Total CPU time [Mh]")

    ax.plot_date([start_date, start_date], [30, 130], ":", lw=lw, color="black")

    ax.plot_date(dates, cpuhs, "o-", lw=lw, label=runName + " local fit")
    ax.plot_date(dates_extra, cpuhs_tng100, "o-", lw=lw, label=runName + " w/ TNG100 cpuh(a)")
    ax.plot_date(dates_extra, cpuhs_1024, "o-", lw=lw, label=runName + " w/ L25n1024 cpuh(a)")

    ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))
    ax.legend()
    fig.autofmt_xdate()

    # (B)
    ax = fig.add_subplot(212)
    ax.set_xlim(xlim_dates)
    ax.set_ylim(ylim_dates)
    ax.set_xlabel("Prediction Date")
    ax.set_ylabel("Estimated Completion Date")

    ax.plot_date(dates, finish_dates, "o-", lw=lw, label=runName + " local fit")
    ax.plot_date(dates_extra, finish_dates_tng100, "o-", lw=lw, label=runName + " w/ TNG100 cpuh(a)")
    ax.plot_date(dates_extra, finish_dates_1024, "o-", lw=lw, label=runName + " w/ L25n1024 cpuh(a)")

    ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))
    ax.yaxis.set_major_formatter(DateFormatter("%b %Y"))
    ax.legend()
    fig.autofmt_xdate()

    fig.savefig(fName1)
    plt.close(fig)


def periodic_slurm_status(machine="vera", nosave=False):
    """Collect current statistics from the SLURM scheduler, save some data, make some plots."""
    import os
    import pwd
    import subprocess

    import h5py
    import pyslurm

    def _expandNodeList(nodeListStr):
        nodesRet = []
        nodeGroups = nodeListStr.split("],")

        for nodeGroup in nodeGroups:
            if "[" not in nodeGroup:  # single node
                nodesRet.append(nodeGroup)
                continue

            if "," in nodeGroup:
                # multiple numeric ranges, e.g. "vera[01-02,101-136]"
                base, num_ranges = nodeGroup.split("[")
                for num_range_str in num_ranges.split(","):
                    num_range = num_range_str.split("-")
                    for num in range(int(num_range[0]), int(num_range[1]) + 1):
                        if len(num_range[0]) == 2:
                            nodesRet.append("%s%02d" % (base, num))
                        if len(num_range[0]) == 3:
                            nodesRet.append("%s%03d" % (base, num))
                continue

            # typical case, e.g. 'freya[01-04]'
            base, num_range = nodeGroup.split("[")
            num_range = num_range[:-1].split("-")
            for num in range(int(num_range[0]), int(num_range[1]) + 1):
                if len(num_range[0]) == 2:
                    nodesRet.append("%s%02d" % (base, num))
                if len(num_range[0]) == 3:
                    nodesRet.append("%s%03d" % (base, num))

        return nodesRet

    # config
    saveDataFile = "historical.hdf5"

    if machine == "freya":
        partNames = ["p.24h", "p.gpu"]
        coresPerNode = 40
        cpusPerNode = 2
        nHyper = 1  # 2 to enable HTing accounting

        rackPrefix = "opasw"
        rackNumberList = [0, 1, 2, 6, 7, 3]
        visRackMultFac = 1  # draw one box per rack

        pos_timeseries_week = [0.838, 0.482, 0.154, 0.15]  # left,bottom,width,height
        pos_series_longterm = [0.67, 0.765, 0.322, 0.154]  # left,bottom,width,height
        numMonthsLongTerm = 6

    if machine == "vera":
        partNames = ["p.vera", "p.large", "p.huge", "p.gpu"]
        coresPerNode = 72
        cpusPerNode = 2
        nHyper = 1  # 2 to enable HTing accounting

        rackPrefix = "ibsw"
        rackNumberList = [0, 1]
        visRackMultFac = 2  # draw 2 boxes per rack

        pos_timeseries_week = [0.507, 0.566, 0.236, 0.14]  # left,bottom,width,height
        pos_series_longterm = [0.507, 0.741, 0.485, 0.142]  # left,bottom,width,height
        numMonthsLongTerm = 4

    allocStates = ["ALLOCATED", "MIXED"]
    idleStates = ["IDLE", "PLANNED"]
    downStates = ["DOWN", "DRAINED", "ERROR", "FAIL", "FAILING", "POWER_DOWN", "UNKNOWN"]

    # get data
    jobs = pyslurm.job().get()
    topo = None  # pyslurm.topology().get() # throwing error
    stats = pyslurm.statistics().get()
    nodes = pyslurm.node().get()
    parts = None  # pyslurm.partition().get() # throwing error

    curTime = datetime.fromtimestamp(stats["req_time"])
    print("Now [%s]." % curTime.strftime("%A (%d %b) %H:%M"))

    # jobs: split, and attach running job info to nodes
    jobs_running = [jobs[jid] for jid in jobs if jobs[jid]["job_state"] == "RUNNING"]
    jobs_pending = [jobs[jid] for jid in jobs if jobs[jid]["job_state"] == "PENDING"]

    for job in jobs_running:
        for nodeName in job["cpus_allocated"]:
            if "cur_job_owner" in nodes[nodeName]:
                print("WARNING: Node [%s] already has a job from [%s]." % (nodeName, nodes[nodeName]["cur_job_owner"]))

            # nodes[nodeName]['cur_job_user'] = subprocess.check_output('id -nu %d'%job['user_id'], shell=True).strip()
            nodes[nodeName]["cur_job_owner"] = pwd.getpwuid(job["user_id"])[4].split(",")[0]
            nodes[nodeName]["cur_job_name"] = job["name"]
            nodes[nodeName]["cur_job_runtime"] = job["run_time_str"]

    n_jobs_running = len(jobs_running)
    n_jobs_pending = len(jobs_pending)

    pending_reasons = [job["state_reason"] for job in jobs_pending]
    n_pending_priority = pending_reasons.count("Priority")
    n_pending_dependency = pending_reasons.count("Dependency")
    n_pending_resources = pending_reasons.count("Resources")
    n_pending_userheld = pending_reasons.count("JobHeldUser")

    if "Resources" in pending_reasons:
        next_job_starting = jobs_pending[pending_reasons.index("Resources")]  # always just 1?
        next_job_starting["user_name"] = pwd.getpwuid(next_job_starting["user_id"])[0]
    else:
        next_job_starting = None

    # restrict nodes to those in main partition (skip login nodes, etc)
    nodesInPart = []
    if parts is not None:
        for partName in partNames:
            nodesInPart += _expandNodeList(parts[partName]["nodes"])
    else:
        nodesInPart = nodes.keys()

    for _, node in nodes.items():
        if node["cpu_load"] == 4294967294:
            node["cpu_load"] = 0  # fix uint32 overflow

    nodes_main = [nodes[name] for name in nodes if name in nodesInPart]
    nodes_misc = [nodes[name] for name in nodes if name not in nodesInPart]

    # nodes: gather statistics
    nodes_idle = []
    nodes_alloc = []
    nodes_down = []

    for node in nodes_main:
        # idle?
        for state in idleStates:
            if state in node["state"]:
                nodes_idle.append(node)
                continue

        # down for any reason?
        for state in downStates:
            if state in node["state"]:
                nodes_down.append(node)
                continue

        # in use?
        for state in allocStates:
            if state in node["state"]:
                nodes_alloc.append(node)
                continue

    # nodes: print statistics
    n_nodes_down = len(nodes_down)
    n_nodes_idle = len(nodes_idle)
    n_nodes_alloc = len(nodes_alloc)

    print(
        " Main nodes: [%d] total, of which [%d] are idle, [%d] are allocated, and [%d] are down."
        % (len(nodes_main), n_nodes_idle, n_nodes_alloc, n_nodes_down)
    )
    print(" Misc nodes: [%d] total." % len(nodes_misc))

    if parts is not None and np.sum(parts[partName]["total_nodes"] for partName in partNames) != len(nodes_main):
        print(" WARNING: Node count mismatch.")
    if len(nodes_main) != n_nodes_idle + n_nodes_alloc + n_nodes_down:
        print(" WARNING: Nodes not all accounted for.")

    nCores = 0
    if parts is not None:
        for partName in partNames:
            nCores += parts[partName]["total_nodes"] * coresPerNode
    else:
        for node in nodes:
            nCores += nodes[node]["cpus"]

    nCores_alloc = np.sum([j["num_cpus"] for j in jobs_running]) / nHyper
    nCores_idle = nCores - nCores_alloc

    print(
        " Cores: [%d] total, of which [%d] are allocated, [%d] are idle or unavailable."
        % (nCores, nCores_alloc, nCores_idle)
    )

    if nCores != nCores_alloc + nCores_idle:
        print(" WARNING: Cores not all accounted for.")

    for node in nodes_main:
        if node["cpu_load"] is None:
            node["cpu_load"] = 0.0

    # cluster: statistics
    cluster_load = float(nCores_alloc) / nCores * 100

    cpu_load_allocnodes_mean = np.mean([float(node["cpu_load"]) / (node["cpus"] / nHyper) for node in nodes_alloc])
    cpu_load_allnodes_mean = np.mean([float(node["cpu_load"]) / (node["cpus"] / nHyper) for node in nodes_main])

    print(
        " Cluster: [%.1f%%] global load, with mean per-node CPU loads: [%.1f%% %.1f%%]."
        % (cluster_load, cpu_load_allocnodes_mean, cpu_load_allnodes_mean)
    )

    # time series data file: create if it doesn't exist already
    nSavePts = 1000000
    saveDataFields = [
        "cluster_load",
        "cpu_load_allocnodes_mean",
        "n_jobs_running",
        "n_jobs_pending",
        "n_nodes_down",
        "n_nodes_idle",
        "n_nodes_alloc",
    ]

    if not os.path.isfile(saveDataFile):
        with h5py.File(saveDataFile, "w") as f:
            for field in saveDataFields:
                f[field] = np.zeros(nSavePts, dtype="float32")
            f["timestamp"] = np.zeros(nSavePts, dtype="int32")
            f.attrs["count"] = 0

    # time series data file: store current data
    if not nosave:
        with h5py.File(saveDataFile, "a") as f:
            ind = f.attrs["count"]
            f["timestamp"][ind] = stats["req_time"]
            for field in saveDataFields:
                f[field][ind] = locals()[field]
            f.attrs["count"] += 1

    # count nodes per rack (disabled: topo plugin hardening issue with slurm 23.11.x)
    # maxNodesPerRack = 0
    # for rackNum in rackNumberList:
    #    rack = topo[rackPrefix + "%d" % (rackNum + 1)]
    #    rackNodes = _expandNodeList(rack["nodes"])
    #    if len(rackNodes) > maxNodesPerRack:
    #        maxNodesPerRack = len(rackNodes)

    # if visRackMultFac > 1:
    #    maxNodesPerRack = int(np.ceil(maxNodesPerRack / visRackMultFac))

    maxNodesPerRack = 36

    # start node figure
    fig = plt.figure(figsize=(18.9, 9.2))
    fig.set_layout_engine("none")  # matplotlib 3.6+

    rackVisBoxes = []  # in case we want to show more than 1 visible box per rack
    for rackNum in rackNumberList:
        for _i in range(visRackMultFac):
            rackVisBoxes.append(rackNum)
    # if machine == 'vera': rackVisBoxes = rackVisBoxes[:-1] # do not split second (small) rack

    for i, rackNum in enumerate(rackVisBoxes):
        if 0:
            # topo plugin hardening issue with slurm 23.11.x
            rack = topo[rackPrefix + "%d" % (rackNum + 1)]  # noqa: F821
            rackNodes = _expandNodeList(rack["nodes"])

            # print(rack['name'], rack['level'], rack['nodes'], len(rackNodes))

            if visRackMultFac > 1 and rackVisBoxes.count(rackNum) > 1:
                # take subset of nodes in this actual rack, to display in this 'virtual rack' box
                segment = i % visRackMultFac
                nPerSeg = int(len(rackNodes) / visRackMultFac)

                # make sure we don't skip any
                if segment == visRackMultFac - 1 and nPerSeg * visRackMultFac != len(rackNodes):
                    rackNodes = rackNodes[nPerSeg * segment :]
                else:
                    # normal case
                    rackNodes = rackNodes[nPerSeg * segment : nPerSeg * (segment + 1)]
        else:
            # vera hard-coded hack
            nodeNames = list(nodes.keys())
            for nn in ["vera01", "vera02"]:  # re-arrange to natural sort
                nodeNames.remove(nn)
                nodeNames.append(nn)
            if i == 0:
                rackNodes = nodeNames[0:36]
            if i == 1:
                rackNodes = nodeNames[36:72]
            if i == 2:
                rackNodes = nodeNames[72:93]
            if i == 3:
                rackNodes = nodeNames[93:]

        ax = fig.add_subplot(1, len(rackVisBoxes), i + 1)
        ax.set_position([i * 0.25 + 0.005, 0.01, 0.24, 0.87])  # left,bottom,width,height

        ax.set_xlim([0, 1])
        ax.set_ylim([-1, maxNodesPerRack])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if len(rackNodes) < maxNodesPerRack:
            # draw shorter rack
            for spine in ["top", "right", "left", "bottom"]:
                ax.spines[spine].set_visible(False)
            ax.plot([0, 1], [-1, -1], "-", lw=1.5, color="black")
            ax.plot([0, 1], [len(rackNodes), len(rackNodes)], "-", lw=1.5, color="black")
            ax.plot([0, 0], [-1, len(rackNodes)], "-", lw=2.0, color="black")
            ax.plot([1, 1], [-1, len(rackNodes)], "-", lw=2.5, color="black")

        # draw representation of each node
        for j, name in enumerate(rackNodes):
            # circle: color by status
            color = "gray"
            if name in [n["name"] for n in nodes_down]:
                color = "red"
            if name in [n["name"] for n in nodes_alloc]:
                color = "green"
            if name in [n["name"] for n in nodes_idle]:
                color = "orange"
            ax.plot(0.14, j, "o", color=color, markersize=10.0)
            textOpts = {"fontsize": 8.0, "horizontalalignment": "left", "verticalalignment": "center"}

            pad = 0.10
            xmin = 0.18
            xmax = 0.475 if machine == "freya" else 0.50
            padx = 0.002
            dx = (xmax - xmin) / (coresPerNode / cpusPerNode)

            maxname = 16 if machine == "freya" else 24

            # entire node
            # ax.fill_between( [xmin,xmax], [j-0.5+pad,j-0.5+pad], [j+0.5-pad, j+0.5-pad], facecolor=color, alpha=0.2)

            # individual cores
            for k in range(cpusPerNode):
                if k == 0:
                    y0 = j - 0.5 + pad
                    y1 = j - pad / 2
                if k == 1:
                    y0 = j + pad / 2
                    y1 = j + 0.5 - pad

                for m in range(int(coresPerNode / cpusPerNode)):
                    ax.fill_between(
                        [xmin + m * dx + padx, xmin + (m + 1) * dx - padx],
                        [y0, y0],
                        [y1, y1],
                        facecolor=color,
                        alpha=0.3,
                    )

            # load
            load = 0.0
            if nodes[name]["cpu_load"] is not None:
                load = float(nodes[name]["cpu_load"]) / (nodes[name]["cpus"] / nHyper)

            if name in [n["name"] for n in nodes_down]:
                load = 0.0  # if down, we don't get an updated value for this

            # color load
            color = "#59a14f" if load > 90.0 else "#e15759"  # green = good (high) usage, red = bad usage
            if load < 1.0 or name in ["vera01", "vera02"]:
                color = "#333333"  # idle
            if name in [n["name"] for n in nodes_down]:
                color = "red"
            ax.text(xmax + padx * 10, j, "%.1f%%" % load, color=color, **textOpts)

            # node name
            ax.text(0.02, j, name.replace("freya", ""), color="#222222", **textOpts)

            if "cur_job_owner" in nodes[name]:
                real_name = nodes[name]["cur_job_owner"]
                real_name = real_name[:maxname] + "..." if len(real_name) > maxname else real_name  # truncate
                ax.text(xmax + 0.14 + padx * 10, j, real_name, color="#333333", **textOpts)
            elif name in [n["name"] for n in nodes_down]:
                ax.text(xmax + 0.14 + padx * 10, j, "down!", color="#333333", **textOpts)

    # time series data load
    data = {}
    with h5py.File(saveDataFile, "r") as f:
        count = f.attrs["count"]
        for key in f.keys():
            data[key] = f[key][0:count]

    # time series plot (last week)
    if 1:
        numDays = 7
        yticks = [60, 70, 80, 90]
        ylim = [50, 100]
        fontsize = 11

        ax = fig.add_axes(pos_timeseries_week)
        # ax.set_ylabel('CPU / Cluster Load [%]')
        ax.set_ylim(ylim)

        minTs = stats["req_time"] - 24 * 60 * 60 * numDays
        w = np.where(data["timestamp"] > minTs)[0]
        dates = [datetime.fromtimestamp(ts) for ts in data["timestamp"] if ts > minTs]

        ax.plot_date(dates, data["cluster_load"][w], "-", label="cluster load")
        ax.plot_date(dates, data["cpu_load_allocnodes_mean"][w], "-", label="<node load>")
        ax.tick_params(axis="y", direction="in", pad=-30)
        ax.yaxis.set_ticks(yticks)
        ax.yaxis.set_ticklabels([str(yt) + "%" for yt in yticks])
        # ax.xaxis.set_major_locator(HourLocator(byhour=[0]))
        ax.xaxis.set_major_formatter(DateFormatter("%a"))  # %Hh
        # ax.xaxis.set_minor_locator(HourLocator(byhour=[12]))
        ax.legend(loc="lower right", fontsize=fontsize)

        for item in [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(fontsize)

    # time series plot (long term e.g. 6 months)
    if 1:
        ax = fig.add_axes(pos_series_longterm)
        ax.set_ylim(ylim)

        minTs = stats["req_time"] - 24 * 60 * 60 * 30 * numMonthsLongTerm
        w = np.where(data["timestamp"] > minTs)[0]
        dates = [datetime.fromtimestamp(ts) for ts in data["timestamp"] if ts > minTs]

        # ax.set_xlim([datetime.datetime(2019,3,1),datetime.datetime(2019,7,12)])

        sKn = 351 if machine == "freya" else 5
        sKo = 3
        data_load1 = savgol_filter(data["cluster_load"][w], sKn, sKo)
        data_load2 = savgol_filter(data["cpu_load_allocnodes_mean"][w], sKn, sKo)

        ax.plot_date(dates, data_load1, "-", label="cluster load")
        ax.plot_date(dates, data_load2, "-", label="<node load>")
        ax.tick_params(axis="y", direction="in", pad=-30)
        ax.yaxis.set_ticks(yticks)
        ax.yaxis.set_ticklabels([str(yt) + "%" for yt in yticks])
        ax.xaxis.set_major_locator(DayLocator(bymonthday=1))  # bymonthday=0
        ax.xaxis.set_major_formatter(DateFormatter("%b"))
        ax.xaxis.set_minor_locator(DayLocator(bymonthday=[7, 14, 21]))
        ax.legend(loc="lower right", fontsize=fontsize)

        for item in [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(fontsize)

    # text
    timeStr = "Last Updated: %s" % curTime.strftime("%a %d %b %H:%M")
    nodesStr = "nodes: [%d] total, of which [%d] are idle, [%d] are allocated, and [%d] are down." % (
        len(nodes_main),
        len(nodes_idle),
        len(nodes_alloc),
        len(nodes_down),
    )
    coresStr = "cores: [%d] total, of which [%d] are allocated, [%d] are idle/unavailable." % (
        nCores,
        nCores_alloc,
        nCores_idle,
    )
    loadStr = "cluster: [%.1f%%] global load, with mean per-node CPU load: [%.1f%%]." % (
        cluster_load,
        cpu_load_allocnodes_mean,
    )
    jobsStr = "jobs: [%d] running, [%d] waiting," % (n_jobs_running, n_pending_priority + n_pending_resources)
    jobsStr2 = "[%d] userheld, & [%d] dependent." % (n_pending_userheld, n_pending_dependency)

    if next_job_starting is not None:
        next_job_starting["name2"] = (
            next_job_starting["name"][:6] + "..." if len(next_job_starting["name"]) > 8 else next_job_starting["name"]
        )  # truncate
        nextJobsStr = "next to run: id=%d %s (%s)" % (
            next_job_starting["job_id"],
            next_job_starting["name2"],
            next_job_starting["user_name"],
        )

    updated_pos = 0.988 if machine == "freya" else 0.735
    title_fs = 48.0 if machine == "freya" else 56.0

    ax.annotate(
        "%s Status" % machine.upper(),
        [0.99, 0.952],
        xycoords="figure fraction",
        fontsize=title_fs,
        ha="right",
        va="center",
    )
    ax.annotate(
        timeStr, [0.99, 0.908], xycoords="figure fraction", fontsize=12.0, ha="right", va="center", color="green"
    )
    ax.annotate(
        nodesStr,
        [0.006, 0.98],
        xycoords="figure fraction",
        fontsize=20.0,
        horizontalalignment="left",
        verticalalignment="center",
    )
    ax.annotate(
        coresStr,
        [0.006, 0.943],
        xycoords="figure fraction",
        fontsize=20.0,
        horizontalalignment="left",
        verticalalignment="center",
    )
    ax.annotate(
        loadStr,
        [0.006, 0.906],
        xycoords="figure fraction",
        fontsize=20.0,
        horizontalalignment="left",
        verticalalignment="center",
    )
    ax.annotate(
        jobsStr,
        [0.71, 0.98],
        xycoords="figure fraction",
        fontsize=20.0,
        horizontalalignment="right",
        verticalalignment="center",
    )
    ax.annotate(
        jobsStr2,
        [0.71, 0.943],
        xycoords="figure fraction",
        fontsize=20.0,
        horizontalalignment="right",
        verticalalignment="center",
    )
    # if next_job_starting is not None:
    #    ax.annotate(nextJobsStr, [0.73, 0.906], xycoords='figure fraction', fontsize=20.0, ha='right', va='center')

    # disk usage text
    df = (
        str(subprocess.check_output("df -h /virgotng /%s/u /%s/ptmp" % (machine, machine), shell=True))
        .replace("b'", "")
        .strip()
        .split("\\n")
    )
    for i, line in enumerate(df):
        if line in ["", "'"]:
            continue
        fsStr = line.split("%")[0] + "%"
        fsStr = fsStr.replace("Size", "   Size")
        fsStr = fsStr.replace("gpfsvirgo", "/virgo/          ")
        fsStr = fsStr.replace("virgotng", "/virgotng/    ")
        fsStr = fsStr.replace("%s_u" % machine, "/%s/u/    " % machine)
        fsStr = fsStr.replace("%s_ptmp" % machine, "/%s/ptmp/" % machine)

        if machine == "freya":
            fs_pos = [0.837, 0.727 - i * 0.026]
            fs_fontsize = 12.5
        if machine == "vera":
            fs_pos = [0.78, 0.70 - i * 0.04]
            fs_fontsize = 16.0

        ax.annotate(
            fsStr,
            fs_pos,
            xycoords="figure fraction",
            fontsize=fs_fontsize,
            horizontalalignment="left",
            verticalalignment="center",
        )

    # save
    fig.savefig("%s_stat_1.png" % machine, dpi=100)  # 1890x920 pixels
    plt.close(fig)
