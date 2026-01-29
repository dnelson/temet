"""
Drivers (i.e. examples) of plots using the generalized functionality.
"""

import glob

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from ..plot import snapshot, subhalos
from ..plot.quantities import quantList
from ..util.simParams import simParams


# --- subhalos ---


def plots():
    """Exploration of 2D histograms, vary over all known quantities as cQuant."""
    sPs = []
    sPs.append(simParams(run="tng100-1", redshift=0.0))

    xQuant = "mgas_r500"
    yQuant = "mstar_r500"  #'mstar30pkpc_mhalo200_ratio' #'ssfr'

    cenSatSelects = ["cen"]  # ['cen','sat','all']
    quants = [None]  # quantList(wTr=False, wMasses=True)

    for sP in sPs:
        for css in cenSatSelects:
            pdf = PdfPages("galaxy_2dhistos_%s_%d_%s_%s_%s.pdf" % (sP.simName, sP.snap, yQuant, xQuant, css))

            for cQuant in quants:
                subhalos.histogram2d(
                    sP,
                    yQuant=yQuant,
                    xQuant=xQuant,
                    xlim=None,  # [6.0, 9.0]
                    ylim=None,  # [-2.5, 0.0] # None
                    clim=None,  # [10.0,11.0]
                    minCount=1,
                    nBins=40,
                    qRestrictions=None,  # [ ['mstar_30pkpc_log',10.0,11.0] ]
                    medianLine=True,
                    cenSatSelect=css,
                    cQuant=cQuant,
                    cStatistic="median_nan",
                    cRel=None,  # [0.6,1.4,False]
                    pdf=pdf,
                )

            pdf.close()


def plots_explore(sP):
    """Exploration of 2D histograms, vary over all known quantities as y-axis."""
    cQuants = [
        "slit_vsigma_halpha",
        "slit_vrot_halpha",
        "slit_voversigma_halpha",
        "slit_vsigma_stars",
        "slit_vrot_stars",
        "slit_voversigma_stars",
    ]

    css = "cen"  # ['cen','sat','all']

    yQuants = quantList(wCounts=False, wTr=False, wMasses=True)

    xQuant = "mstar_30pkpc_log"
    xlim = [8.7, 11.2]

    for cQuant in cQuants:
        pdf = PdfPages("2dhistos_%s_%d_x=%s_y=all_c=%s_%s.pdf" % (sP.simName, sP.snap, xQuant, cQuant, css))

        for yQuant in yQuants:
            subhalos.histogram2d(sP, yQuant=yQuant, xQuant=xQuant, xlim=xlim, cenSatSelect=css, cQuant=cQuant, pdf=pdf)

        pdf.close()


def plots2():
    """Exploration of 1D slices."""
    sPs = []
    sPs.append(simParams(res=1820, run="tng", redshift=0.0))
    sPs.append(simParams(res=2500, run="tng", redshift=0.0))

    xQuant = "color_C_gr"
    sQuant = "mstar2_log"
    sRange = [10.4, 10.6]
    cenSatSelects = ["cen"]

    quants = quantList(wCounts=False, wTr=False)
    quantsTr = quantList(wCounts=False, onlyTr=True)

    for css in cenSatSelects:
        pdf = PdfPages(
            "galaxyColor_1Dslices_%s_%s_%s-%.1f-%.1f_%s.pdf"
            % ("-".join([sP.simName for sP in sPs]), xQuant, sQuant, sRange[0], sRange[1], css)
        )

        # all quantities on one multi-panel page:
        # subhalos.slice(sPs, xQuant=xQuant, yQuants=quants, sQuant=sQuant,
        #  sRange=sRange, cenSatSelect=css, pdf=pdf)
        # subhalos.slice(sPs, xQuant=xQuant, yQuants=quantsTr, sQuant=sQuant,
        #  sRange=sRange, cenSatSelect=css, pdf=pdf)

        # one page per quantity:
        for yQuant in quants + quantsTr:
            subhalos.slice(
                sPs, xQuant=xQuant, yQuants=[yQuant], sQuant=sQuant, sRange=sRange, cenSatSelect=css, pdf=pdf
            )

        pdf.close()


def plots3():
    """Exploration of median trends."""
    sPs = []
    sPs.append(simParams(res=1820, run="tng", redshift=0.0))
    # sPs.append( simParams(res=1820, run='illustris', redshift=0.0) )
    # sPs.append( simParams(res=2500, run='tng', redshift=0.0) )

    xQuant = "mstar_30pkpc"  #'mhalo_200_log',mstar1_log','mstar_30pkpc'
    cenSatSelects = ["all"]

    sQuant = "color_C_gr"  #'mstar_out_100kpc_frac_r200'
    sLowerPercs = [10, 50]
    sUpperPercs = [90, 50]

    yQuants = quantList(wCounts=False, wTr=True, wMasses=True)
    yQuants = ["size_stars"]

    # make plots
    for css in cenSatSelects:
        pdf = PdfPages(
            "medianQuants_%s_x=%s_%s_slice=%s.pdf" % ("-".join([sP.simName for sP in sPs]), xQuant, css, sQuant)
        )

        # all quantities on one multi-panel page:
        subhalos.median(
            sPs,
            yQuants=yQuants,
            xQuant=xQuant,
            cenSatSelect=css,
            sQuant=sQuant,
            sLowerPercs=sLowerPercs,
            sUpperPercs=sUpperPercs,
            pdf=pdf,
        )

        # individual plot per y-quantity:
        # for yQuant in yQuants:
        #    subhalos.median(sPs, yQuants=[yQuant], xQuant=xQuant, cenSatSelect=css,
        #      sQuant=sQuant, sLowerPercs=sLowerPercs, sUpperPercs=sUpperPercs, pdf=pdf)

        # individual plot per s-quantity:
        # for sQuant in sQuants:
        #    subhalos.median(sPs, yQuants=yQuant, xQuant=xQuant, cenSatSelect=css,
        #      sQuant=[sQuant], sLowerPercs=sLowerPercs, sUpperPercs=sUpperPercs, pdf=pdf)

        pdf.close()


def plots4():
    """Single median trend."""
    sPs = []
    # sPs.append( simParams(run='tng50-1, redshift=2.0) )
    sPs.append(simParams(run="tng100-1", redshift=0.0))

    xQuant = "mstar_30pkpc"  #'mhalo_200_log',mstar1_log','mstar_30pkpc'
    yQuant = "xray_05_2kev_r500"
    scatterColor = "ssfr"  #'size_gas' #'M_bulge_counter_rot' # 'size_stars'
    cenSatSelect = "cen"
    filterFlag = False  # True

    xlim = [9.0, 11.5]  # [10.2,11.6]
    ylim = None  # [4.7,1.5]
    clim = [-2.0, -0.5]  # [1.0, 2.0]
    scatterPoints = True
    drawMedian = True
    markersize = 20.0
    maxPointsPerDex = 2000

    sQuant = None  #'color_C_gr'
    # sLowerPercs = None  # [10,50]
    # sUpperPercs = None  # [90,50]

    qRestrictions = None
    # qRestrictions = [ ['delta_sfms',-0.5,np.inf] ] #  [ ['mstar_30pkpc_log',10.0,11.0] ] # SINS-AO rough cut

    pdf = PdfPages(
        "median_x=%s_y=%s_%s_slice=%s_%s_z%.1f.pdf"
        % (xQuant, yQuant, cenSatSelect, sQuant, sPs[0].simName, sPs[0].redshift)
    )

    # one quantity
    subhalos.median(
        sPs,
        yQuants=[yQuant],
        xQuant=xQuant,
        cenSatSelect=cenSatSelect,
        # sQuant=sQuant, sLowerPercs=sLowerPercs, sUpperPercs=sUpperPercs,
        qRestrictions=qRestrictions,
        xlim=xlim,
        ylim=ylim,
        clim=clim,
        drawMedian=drawMedian,
        markersize=markersize,
        scatterPoints=scatterPoints,
        scatterColor=scatterColor,
        maxPointsPerDex=maxPointsPerDex,
        markSubhaloIDs=None,
        filterFlag=filterFlag,
        pdf=pdf,
    )

    pdf.close()


def plots5():
    """Single median trend over multiple redshifts."""
    sPs = []
    # sPs.append( simParams(res=2160, run='tng', redshift=2.0) )
    sPs.append(simParams(res=1820, run="tng", redshift=3.0))
    sPs.append(simParams(res=1820, run="tng", redshift=2.0))
    sPs.append(simParams(res=1820, run="tng", redshift=1.0))
    sPs.append(simParams(res=1820, run="tng", redshift=0.0))
    # sPs.append( simParams(res=2500, run='tng', redshift=1.0) )

    xQuant = "size_stars"  # mstar_30pkpc' #'mhalo_200_log',mstar1_log','mstar_30pkpc'
    yQuant = "fdm1"
    scatterColor = None  #'mstar_30pkpc' #'M_bulge_counter_rot' # 'size_stars'
    cenSatSelect = "cen"
    filterFlag = False  # True

    xlim = [-0.4, 1.5]  # [9.0, 11.5] #[10.2,11.6]
    ylim = None  # [4.7,1.5]
    scatterPoints = False
    drawMedian = True
    markersize = None  # 20.0
    maxPointsPerDex = None  # 2000

    clim = [10, 11.2]

    sQuant = None  #'color_C_gr'

    qRestrictions = [["mstar_30pkpc_log", 10.0, 11.0], ["delta_sfms", -0.5, np.inf]]  # SINS-AO rough cut

    pdf = PdfPages(
        "median_x=%s_y=%s_%s_slice=%s_%s_z%.1f.pdf"
        % (xQuant, yQuant, cenSatSelect, sQuant, sPs[0].simName, sPs[0].redshift)
    )

    # one quantity
    subhalos.median(
        sPs,
        yQuants=[yQuant],
        xQuant=xQuant,
        cenSatSelect=cenSatSelect,
        qRestrictions=qRestrictions,
        xlim=xlim,
        ylim=ylim,
        clim=clim,
        drawMedian=drawMedian,
        markersize=markersize,
        scatterPoints=scatterPoints,
        scatterColor=scatterColor,
        maxPointsPerDex=maxPointsPerDex,
        markSubhaloIDs=None,
        filterFlag=filterFlag,
        pdf=pdf,
    )

    pdf.close()


def plots_uvj():
    """Explore UVJ color-color diagram."""
    sPs = []
    sPs.append(simParams(res=1820, run="tng", redshift=0.0))
    # sPs.append( simParams(res=2500, run='tng', redshift=2.0) )

    yQuant = "color_C-30kpc-z_UV"  #'color_nodust_UV' #
    xQuant = "color_C-30kpc-z_VJ"  #'color_nodust_VJ' #
    cenSatSelects = ["all"]  # ['cen','sat','all']
    pStyle = "white"

    cNaNZeroToMin = True  # False
    medianLine = False  # True

    if 0:
        # color-coded by SFR
        cs = "median_nan"
        quants = ["ssfr"]
        clim = [-2.5, 0.0]  # None
        minCount = 10
        qRestrictions = None
        xlim = None
        ylim = None
    if 1:
        cs = "count"
        quants = [None]
        clim = [0.0, 2.5]  # log N_gal
        minCount = 0
        qRestrictions = [["mstar_30pkpc_log", 10.0, np.inf]]  # LEGA-C mass cut
        xlim = [0.0, 1.7]
        ylim = [0.4, 2.6]

    for sP in sPs:
        for css in cenSatSelects:
            pdf = PdfPages("galaxy_2dhistos_%s_%d_%s_%s_%s_%s.pdf" % (sP.simName, sP.snap, yQuant, xQuant, cs, css))

            for cQuant in quants:
                subhalos.histogram2d(
                    sP,
                    yQuant=yQuant,
                    xQuant=xQuant,
                    xlim=xlim,
                    ylim=ylim,
                    clim=clim,
                    cNaNZeroToMin=cNaNZeroToMin,
                    minCount=minCount,
                    medianLine=medianLine,
                    cenSatSelect=css,
                    cQuant=cQuant,
                    cStatistic=cs,
                    qRestrictions=qRestrictions,
                    pStyle=pStyle,
                    pdf=pdf,
                )

            pdf.close()


def plots_tng50_structural(rel=False, sP=None):
    """Exploration of 2D histograms."""
    if sP is None:
        sP = simParams(res=2160, run="tng", redshift=1.0)

    xQuant = "mstar_30pkpc_log"
    xlim = [8.7, 11.2]
    yQuants = [
        "slit_vsigma_halpha",
        "slit_vrot_halpha",
        "slit_voversigma_halpha",
        "slit_vsigma_starlight",
        "slit_vrot_starlight",
        "slit_voversigma_starlight",
    ]

    quants_gas = [
        None,
        "sfr2",
        "ssfr",
        "delta_sfms",
        "fgas2_alt",
        "etaM_100myr_20kpc_0kms",
        "vout_90_20kpc",
        "size2d_halpha",
        "diskheight2d_halpha",
        "diskheightnorm2d_halpha",
        "shape_s_sfrgas",
        "shape_ratio_sfrgas",
    ]
    quants_stars = [
        "size2d_starlight",
        "diskheight2d_starlight",
        "diskheightnorm2d_starlight",
        "shape_s_stars",
        "shape_ratio_stars",
    ]

    css = "cen"
    clim = None
    nBins = 60

    if rel:
        cRel = [0.5, 1.5, False]  # [cMin,cMax,cLog] #None
    else:
        cRel = None

    for yQuant in yQuants:
        pdf = PdfPages("2dhisto_%s_%d_x=%s_y=%s_rel=%s_%s.pdf" % (sP.simName, sP.snap, xQuant, yQuant, rel, css))

        for cQuant in quants_gas + quants_stars + yQuants:
            if cQuant == yQuant:
                continue

            subhalos.histogram2d(
                sP,
                yQuant=yQuant,
                xQuant=xQuant,
                xlim=xlim,
                clim=clim,
                cenSatSelect=css,
                cQuant=cQuant,
                nBins=nBins,
                cRel=cRel,
                pdf=pdf,
            )
        pdf.close()

    # return with all cached data, can be passed back in for rapid re-plotting
    return sP


# --- snapshot ---


def compareRuns_PhaseDiagram():
    """Compare a series of runs in a PDF booklet of phase diagrams."""
    # config
    redshift = 0.0
    yQuant = "temp"
    xQuant = "numdens"

    # get list of all 512 method runs via filesystem search
    sP = simParams(res=512, run="tng", redshift=redshift, variant="0000")
    dirs = glob.glob(sP.arepoPath + "../L25n512_*")
    variants = sorted([d.rsplit("_", 1)[1] for d in dirs])
    variants = ["0000", "1006"]

    # start PDF, add one page per run
    pdf = PdfPages("compareRunsPhaseDiagram.pdf")

    for variant in variants:
        sP = simParams(res=512, run="tng", redshift=redshift, variant=variant)
        if sP.simName == "DM only":
            continue
        print(variant, sP.simName)
        snapshot.phaseSpace2d(sP, xQuant=xQuant, yQuant=yQuant, pdf=pdf)

    pdf.close()


def oneRun_PhaseDiagram(redshift=None, snaps=None, hInd=10677, res=13, variant="ST7"):
    """Density temperature phase diagram for a single halo or full box, over multiple snapshots."""
    # config
    sim = simParams(run="structures", hInd=hInd, res=res, variant=variant, redshift=redshift)

    yQuant = "temp"
    xQuant = "nh"

    if sim.isZoom:
        xlim = [-7.0, 5.0]
        ylim = [1.0, 7.0]
        haloIDs = None  # [0]
        qRestrictions = [["rad_rvir", 0.0, 5.0]]  # None #
        clim = [-2.0, -0.2]
    else:
        xlim = [-9.0, 2.0]
        ylim = [2.0, 8.5]
        clim = [-6.0, -0.2]
        haloIDs = None  # full box

    # single snapshot, or multiple?
    if redshift is None and snaps is None:
        snaps = sim.validSnapList()[::10]  # [99]
    if redshift is not None:
        snaps = [sim.snap]

    # start PDF, add one page per snapshot
    for snap in snaps:
        sim.setSnap(snap)

        pdf = PdfPages("phaseDiagram_%s_%s_%d.pdf" % (yQuant, sim.simName, snap))

        snapshot.phaseSpace2d(
            sim,
            xQuant=xQuant,
            yQuant=yQuant,
            haloIDs=haloIDs,
            qRestrictions=qRestrictions,
            xlim=xlim,
            ylim=ylim,
            clim=clim,
            hideBelow=False,
            pdf=pdf,
        )

        pdf.close()


def oneRun_tempcheck():
    """Driver."""
    # config
    sP = simParams(run="tng50-1")
    xQuant = "nh"

    xlim = [-9.0, 3.0]
    ylim = [1.0, 8.5]
    clim = [-6.0, -0.2]

    snaps = sP.validSnapList()

    # start PDF, add one page for temp, one for old_temp
    for snap in snaps:
        sP.setSnap(snap)
        print(snap)

        pdf = PdfPages("phaseCheck_%s_%d.pdf" % (sP.simName, snap))

        snapshot.phaseSpace2d(
            sP, xQuant=xQuant, yQuant="temp", xlim=xlim, ylim=ylim, clim=clim, hideBelow=False, pdf=pdf
        )
        snapshot.phaseSpace2d(
            sP, xQuant=xQuant, yQuant="temp_old", xlim=xlim, ylim=ylim, clim=clim, hideBelow=False, pdf=pdf
        )

        pdf.close()


def compareRuns_RadProfiles():
    """Compare median radial profile of a quantity, differentiating between two runs."""
    variants = ["0000", "0010"]

    sPs = []
    subhaloIDs = []

    for variant in variants:
        sPs.append(simParams(res=512, run="tng", redshift=0.0, variant=variant))

        mhalo = sPs[-1].groupCat(fieldsSubhalos=["mhalo_200_log"])
        with np.errstate(invalid="ignore"):
            w = np.where((mhalo > 11.5) & (mhalo < 12.5))

        subhaloIDs.append(w[0])

    for field in ["temp"]:  # ,'dens','P_gas','z_solar']:
        snapshot.profilesStacked1d(sPs, subhaloIDs=subhaloIDs, ptType="gas", ptProperty=field, weighting="O VI mass")


def compareHaloSets_RadProfiles():
    """Compare median radial profile of a quantity, differentiating between two different types of halos. One run."""
    sPs = []
    sPs.append(simParams(res=1820, run="tng", redshift=2.0))
    # sPs.append( simParams(res=1820,run='tng',redshift=2.0) )
    # sPs.append( simParams(res=1820,run='tng',redshift=2.0) )

    # select subhalos
    mhalo = sPs[0].groupCat(fieldsSubhalos=["mhalo_200_log"])

    if 0:
        gr, _, _, _ = sPs[0].simSubhaloQuantity("color_B_gr")

        with np.errstate(invalid="ignore"):
            w1 = np.where((mhalo > 11.8) & (mhalo < 12.2) & (gr < 0.35))
            w2 = np.where((mhalo > 11.8) & (mhalo < 12.2) & (gr > 0.65))

        print(len(w1[0]), len(w2[0]))

        subhaloIDs = [
            {r"11.8 < M$_{\rm halo}$ < 12.2, (g-r) < 0.35": w1[0], r"11.8 < M$_{\rm halo}$ < 12.2, (g-r) > 0.65": w2[0]}
        ]

    if 1:
        with np.errstate(invalid="ignore"):
            # w0 = np.where((mhalo > 11.3) & (mhalo < 13.4))
            w1 = np.where((mhalo > 11.9) & (mhalo < 12.1))
            w2 = np.where((mhalo > 12.2) & (mhalo < 12.4))
            # w3 = np.where((mhalo > 12.5) & (mhalo < 12.7))

        # subhaloIDs = [{r'M$_{\rm halo}$ = 12.0':w1[0]},
        #            {r'M$_{\rm halo}$ = 12.3':w2[0]},
        #            {r'M$_{\rm halo}$ = 12.6':w3[0]}]
        subhaloIDs = [{r"M$_{\rm halo}$ = 12.0": w1[0]}]
        # subhaloIDs = [{r'M$_{\rm halo}$ broad':w0[0]}]

    # select properties
    ptType = "dm"  # 'gas'
    # fields = ['tcool'] #['metaldens','dens','temp','P_gas','z_solar']
    fields = ["mass"]
    weighting = None  #'O VI mass'
    op = "sum"
    plotIndiv = True

    # proj2D = [2, None] # z-axis, no depth restriction
    proj2D = [2, 10.0]  # z-axis, 10 code units depth = 10 pkpc at z=2

    for field in fields:
        snapshot.profilesStacked1d(
            sPs,
            subhaloIDs=subhaloIDs,
            ptType=ptType,
            ptProperty=field,
            op=op,
            weighting=weighting,
            proj2D=proj2D,
            plotIndiv=plotIndiv,
        )


def compareHaloSets_1DHists():
    """Compare 1D histograms of a quantity, overplotting several halos. One run."""
    sPs = []

    sPs.append(simParams(res=910, run="tng", redshift=0.0))
    mhalo = sPs[-1].groupCat(fieldsSubhalos=["mhalo_200_log"])
    with np.errstate(invalid="ignore"):
        w1 = np.where((mhalo > 11.8) & (mhalo < 12.2))
    subhaloIDs = [w1[0][0:5]]

    if 0:
        # add a second run
        sPs.append(simParams(res=455, run="tng", redshift=0.0))
        mhalo = sPs[-1].groupCat(fieldsSubhalos=["mhalo_200_log"])
        with np.errstate(invalid="ignore"):
            w2 = np.where((mhalo > 11.8) & (mhalo < 12.2))
        subhaloIDs.append(w2[0][0:5])

    for field in ["temp"]:  # ['tcool','vrad']:
        snapshot.histogram1d(sPs, subhaloIDs=subhaloIDs, ptType="gas", ptProperty=field)


def singleHaloProperties():
    """Several phase/radial profile plots for a single halo."""
    sP = simParams(res=256, run="tng", redshift=0.0, variant="0000")

    partType = "gas"
    xQuant = "coolrate"
    yQuant = "dens"

    # pick a MW
    gc = sP.groupCat(fieldsHalos=["Group_M_Crit200", "GroupPos"])
    haloMasses = sP.units.codeMassToLogMsun(gc["Group_M_Crit200"])

    haloIDs = np.where((haloMasses > 12.02) & (haloMasses < 12.03))[0]

    rMin = None
    rMax = None

    snapshot.median(
        [sP], partType=partType, xQuant=xQuant, yQuant=yQuant, haloIDs=haloIDs, radMinKpc=rMin, radMaxKpc=rMax
    )


def compareRuns_particleQuant():
    """Compare a series of runs in a single panel plot of a particle median quantity vs another."""
    # config
    yQuant = "tcool"
    xQuant = "dens_critb"
    ptType = "gas"
    variants = ["0000"]  # ,'2102','2202','2302']

    # start PDF, add one page per run
    sPs = []
    for variant in variants:
        sP = simParams(res=512, run="tng", redshift=0.0, variant=variant)
        if sP.simName == "DM only":
            continue
        sPs.append(sP)

    snapshot.median(sPs, partType=ptType, xQuant=xQuant, yQuant=yQuant)


def coolingPhase():
    """Phase diagram colored by different quantities."""
    # config
    yQuant = "temp"
    xQuant = "numdens"
    cQuants = ["coolrate", "heatrate", "coolrate_powell"]
    xlim = [-9.0, 2.0]
    ylim = [2.0, 8.0]

    # sP = simParams(res=256,run='tng',redshift=0.0,variant='0000')
    sP = simParams(res=1820, run="tng", redshift=0.0)

    # start PDF, add one page per run
    pdf = PdfPages("phaseDiagram_B_%s_%d.pdf" % (sP.simName, sP.snap))

    for cQuant in cQuants:
        snapshot.phaseSpace2d(
            sP,
            xQuant=xQuant,
            yQuant=yQuant,
            meancolors=[cQuant],
            xlim=xlim,
            ylim=ylim,
            weights=None,
            hideBelow=False,
            haloIDs=None,
            pdf=pdf,
        )

    pdf.close()
