"""
Visualizations: common routines.
"""

from os import makedirs
from os.path import expanduser

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter

from ..plot.util import loadColorTable, setAxisColors
from ..util.boxRemap import remapPositions
from ..util.helper import pSplitRange
from ..util.rotation import rotateCoordinateArray
from ..vis.quantities import gridOutputProcess
from ..vis.render import defaultHsmlFac, gridBox


# all frames output here (current directory if empty string)
savePathDefault = expanduser("~") + "/"  # for testing/quick outputs
savePathBase = expanduser("~") + "/data/frames/"  # for large outputs


def addBoxMarkers(p, conf, ax, pExtent):
    """Factor out common annotation/markers to overlay."""

    def _addCirclesHelper(
        p, ax, pos, radii, numToAdd, labelVals=None, lw=1.5, alpha=0.3, marker="o", color="#ffffff", facecolor=None
    ):
        """Helper function to add a number of circle markers for halos/subhalos/SMBHs, within the panel."""
        fontsize = 16  # for text only

        circOpts = {"alpha": alpha, "linewidth": lw}
        textOpts = {
            "color": color,
            "alpha": alpha,
            "fontsize": fontsize,
            "horizontalalignment": "left",
            "verticalalignment": "center",
        }

        if facecolor is None:
            circOpts["fill"] = False
            circOpts["color"] = color
        else:
            circOpts["fill"] = True
            circOpts["edgecolor"] = color
            circOpts["facecolor"] = facecolor

        countAdded = 0
        gcInd = 0

        if p["rotMatrix"] is not None:
            rotCenter = p["rotCenter"]
            if rotCenter is None:
                # use subhalo center at this snapshot
                sh = p["sP"].groupCatSingle(subhaloID=p["sP"].subhaloInd)
                rotCenter = sh["SubhaloPos"]

                if not p["sP"].isZoom and p["sP"].subhaloInd is None:
                    raise Exception("Rotation in periodic box must be about a halo center.")

            pos, _ = rotateCoordinateArray(p["sP"], pos, p["rotMatrix"], rotCenter)

        if pos.ndim == 1 and pos.size == 3:
            assert radii.size == 1
            pos = np.reshape(pos, (1, 3))
            radii = np.array([radii])

        # remap? transform coordinates
        if "remapRatio" in p and p["remapRatio"] is not None:
            pos, _ = remapPositions(p["sP"], pos, p["remapRatio"], p["nPixels"])

        if str(numToAdd) == "all":
            numToAdd = pos.shape[0]

        while countAdded < numToAdd:
            xyzPos = pos[gcInd, :][[p["axes"][0], p["axes"][1], 3 - p["axes"][0] - p["axes"][1]]]
            xyzDist = xyzPos - p["boxCenter"]
            p["sP"].correctPeriodicDistVecs(xyzDist)
            xyzDistAbs = np.abs(xyzDist)

            # in bounds?
            if (
                (xyzDistAbs[0] <= p["boxSizeImg"][0] / 2)
                & (xyzDistAbs[1] <= p["boxSizeImg"][1] / 2)
                & (xyzDistAbs[2] <= p["boxSizeImg"][2] / 2)
                & (radii[gcInd] > 0)
            ):
                # draw and count
                countAdded += 1

                xPos = pos[gcInd, p["axes"][0]]
                yPos = pos[gcInd, p["axes"][1]]
                rad = radii[gcInd] * 1.0

                # our plot coordinate system is true simulation coordinates, except without
                # any periodicity, e.g. relative to boxCenter but restored (negatives or >boxSize ok)
                if xPos > p["extent"][1]:
                    xPos -= p["boxSizeImg"][0]
                if yPos > p["extent"][3]:
                    yPos -= p["boxSizeImg"][1]
                if xPos < p["extent"][0]:
                    xPos += p["boxSizeImg"][0]
                if yPos < p["extent"][2]:
                    yPos += p["boxSizeImg"][1]

                if "relCoords" in p and p["relCoords"]:
                    xPos = xyzDist[0]
                    yPos = xyzDist[1]

                if p["axesUnits"] == "kpc":
                    xPos = p["sP"].units.codeLengthToKpc(xPos)
                    yPos = p["sP"].units.codeLengthToKpc(yPos)
                    rad = p["sP"].units.codeLengthToKpc(rad)
                if p["axesUnits"] == "mpc":
                    xPos = p["sP"].units.codeLengthToMpc(xPos)
                    yPos = p["sP"].units.codeLengthToMpc(yPos)
                    rad = p["sP"].units.codeLengthToMpc(rad)
                assert p["axesUnits"] not in ["deg", "arcsec", "arcmin"]  # todo

                if marker == "o":
                    c = plt.Circle((xPos, yPos), rad, **circOpts)
                    ax.add_artist(c)
                elif marker == "x":
                    # note: markeredgewidth = 0 is matplotlibrc default, need to override
                    ax.plot(
                        xPos,
                        yPos,
                        marker="x",
                        markersize=lw * 4,
                        markeredgecolor=color,
                        markeredgewidth=lw,
                        alpha=alpha,
                    )

                # add text annotation?
                if labelVals is not None:
                    # construct string, labelVals is a dictionary of (k,v) where k is a
                    # format string, and v is a ndarray of values for the string, one per object
                    text = ""
                    for key in labelVals.keys():
                        text += key % labelVals[key][gcInd] + "\n"
                    text = text.strip()

                    # draw text string
                    xPosText = xPos + rad + p["boxSizeImg"][0] / 200
                    yPosText = yPos
                    ax.text(xPosText, yPosText, text, **textOpts)

            gcInd += 1
            if gcInd >= pos.shape[0] and countAdded < numToAdd:
                if numToAdd != pos.shape[0]:
                    # only exactly equal if numToAdd was 'all'
                    print("Warning: Ran out of halos/objects to add, only [%d of %d]" % (countAdded, numToAdd))
                break

        # special behavior: highlight the progenitor of a specific object
        if 0:
            sP_loc = p["sP"].copy()
            sP_loc.setRedshift(0.0)
            mpb = sP_loc.loadMPB(585369)  # Christoph Saulder boundary object

            w = np.where(mpb["SnapNum"] == p["sP"].snap)[0]
            xyzpos = np.squeeze(mpb["SubhaloPos"][w, :])
            rad = 50.0

            c = plt.Circle(
                (xyzpos[p["axes"][0]], xyzpos[p["axes"][1]]), rad, color="red", alpha=alpha, linewidth=2.0, fill=False
            )
            ax.add_artist(c)

        # special heavior: highlight a specific set of (sub)halo inds
        if 0:
            halo_inds = [0, 1, 2, 5]
            for halo_ind in halo_inds:
                halo_loc = p["sP"].halo(halo_ind)
                xyzpos = halo_loc["GroupPos"]
                rad = halo_loc["Group_R_Crit200"] * 2.0
                c = plt.Circle(
                    (xyzpos[p["axes"][0]], xyzpos[p["axes"][1]]),
                    rad,
                    color="red",
                    alpha=alpha,
                    linewidth=2.0,
                    fill=False,
                )
                ax.add_artist(c)

        # special behavior: visualize PMGRID cells next to a periodic boundary
        if 0:
            PMGRID = p["sP"].snapConfigVars()["PMGRID"]  # 4096
            gridSizeCode = p["sP"].boxSize / PMGRID

            ax.plot([0, p["sP"].boxSize], [0, 0], "-", lw=1.0, color="orange", alpha=0.8)
            ax.plot([0, p["sP"].boxSize], [gridSizeCode, gridSizeCode], "-", lw=1.0, color="orange", alpha=0.8)
            ax.plot([0, p["sP"].boxSize], [-gridSizeCode, -gridSizeCode], "--", lw=1.0, color="orange", alpha=0.8)

    if "plotHalos" in p and p["plotHalos"] > 0:
        # plotting N most massive halos in visible area
        sP_load = p["sP"] if not p["sP"].isSubbox else p["sP"].parentBox

        h = sP_load.groupCatHeader()

        if h["Ngroups_Total"] > 0:
            gc = sP_load.groupCat(fieldsHalos=["GroupPos", "Group_R_Crit200"])

            labelVals = None
            if "labelHalos" in p and p["labelHalos"]:
                # label N most massive halos with some properties
                gc_h = sP_load.groupCat(fieldsHalos=["GroupFirstSub", "Group_M_Crit200"])
                halo_mass_logmsun = sP_load.units.codeMassToLogMsun(gc_h["Group_M_Crit200"])
                halo_id = np.arange(gc_h["GroupFirstSub"].size)
                gc_s = sP_load.groupCat(fieldsSubhalos=["mstar_30pkpc_log"])
                sub_ids = gc_h["GroupFirstSub"]

                # construct dictionary of properties (one or more)
                labelVals = {}
                if "mstar" in p["labelHalos"]:  # label with M*
                    labelVals[r"M$_\star$ = 10$^{%.1f}$ M$_\odot$"] = gc_s[sub_ids]
                # if 'mhalo' in p['labelHalos']: # label with M200
                #    labelVals[r'M$_{\rm h}$ = 10$^{%.1f}$ M$_\odot$'] = halo_mass_logmsun
                if "mhalo" in p["labelHalos"]:  # label with M200
                    labelVals[r"%.1f"] = halo_mass_logmsun
                if "id" in p["labelHalos"]:
                    labelVals["[%d]"] = halo_id

            _addCirclesHelper(p, ax, gc["GroupPos"], gc["Group_R_Crit200"], p["plotHalos"], labelVals, alpha=0.5)

    if "plotSubhalos" in p and (str(p["plotSubhalos"]) == "all" or p["plotSubhalos"] > 0):
        # plotting N most massive child subhalos in visible area
        h = p["sP"].groupCatHeader()

        if h["Ngroups_Total"] > 0:
            haloInd = p["sP"].groupCatSingle(subhaloID=p["subhaloInd"])["SubhaloGrNr"]
            halo = p["sP"].groupCatSingle(haloID=haloInd)

            if halo["GroupFirstSub"] != p["subhaloInd"]:
                print("Warning: Rendering subhalo circles around a non-central subhalo!")

            subInds = np.arange(halo["GroupFirstSub"] + 1, halo["GroupFirstSub"] + halo["GroupNsubs"])

            gc = p["sP"].groupCat(fieldsSubhalos=["SubhaloPos", "SubhaloHalfmassRad"])
            gc["SubhaloPos"] = gc["SubhaloPos"][subInds, :]
            gc["SubhaloHalfmassRad"] = gc["SubhaloHalfmassRad"][subInds]

            _addCirclesHelper(p, ax, gc["SubhaloPos"], gc["SubhaloHalfmassRad"], p["plotSubhalos"])

    if "plotBHs" in p and (str(p["plotBHs"]) == "all" or p["plotBHs"] > 0):
        # plotting N most massive PartType5 in visible area
        if p["sP"].numPart[p["sP"].ptNum("bhs")] > 0:
            # global load entire snapshot
            smbh_pos = p["sP"].snapshotSubset("bhs", "pos")
            smbh_mass = p["sP"].snapshotSubset("bhs", "BH_Mass")

            # simple size scaling for visibility: 2px + 1px per dex of log(m_smbh)
            pxScale = p["boxSizeImg"][p["axes"][0]] / 1000
            smbh_rad = (3 + p["sP"].units.codeMassToLogMsun(smbh_mass)) * pxScale

            _addCirclesHelper(p, ax, smbh_pos, smbh_rad, p["plotBHs"], alpha=0.7, lw=3, color="#fff", facecolor="#000")

    if "plotHaloIDs" in p:
        # plotting halos/groups specified by ID, in visible area
        haloInds = p["plotHaloIDs"]
        gc = p["sP"].groupCat(fieldsHalos=["GroupPos", "Group_R_Crit200"])
        gc["GroupPos"] = gc["GroupPos"][haloInds, :]
        rad = 10.0 * gc["Group_R_Crit200"][haloInds]
        labelVals = {"%d": p["plotHaloIDs"]}  # label IDs

        if p["sP"].groupCatHasField("Group", "GroupOrigHaloID"):
            GroupOrigHaloID = p["sP"].halos("GroupOrigHaloID")
            labelVals = {"%d": GroupOrigHaloID[p["plotHaloIDs"]]}

        _addCirclesHelper(p, ax, gc["GroupPos"], rad, len(p["plotHaloIDs"]), labelVals)

    if "plotSubhaloIDs" in p:
        # plotting child subhalos specified by ID, in visible area
        subInds = p["plotSubhaloIDs"]
        gc = p["sP"].groupCat(fieldsSubhalos=["SubhaloPos", "SubhaloHalfmassRadType"])
        gc["SubhaloPos"] = gc["SubhaloPos"][subInds, :]
        rad = 20.0 * gc["SubhaloHalfmassRadType"][subInds, 4]
        labelVals = {"%d": p["plotSubhaloIDs"]}  # label IDs

        _addCirclesHelper(p, ax, gc["SubhaloPos"], rad, len(p["plotSubhaloIDs"]), labelVals)

    if "customCircles" in p:
        # plotting custom list of (x,y,z),(rad) inputs as circles, inputs in simdata coordinates
        _addCirclesHelper(
            p,
            ax,
            p["customCircles"]["pos"],
            p["customCircles"]["rad"],
            p["customCircles"]["rad"].size,
            lw=1.0,
            alpha=0.7,
        )

    if "customCrosses" in p:
        # plotting custom list of (x,y,z) inputs as crosses, inputs in simdata coordinates
        nPoints = p["customCrosses"]["pos"].shape[0]
        _addCirclesHelper(p, ax, p["customCrosses"]["pos"], np.ones(nPoints), nPoints, lw=1.0, alpha=0.8, marker="x")

    if "drawFOV" in p:
        # draw a square 'field of view' in the center of the image, input in arcseconds
        arcsec = p["drawFOV"]
        fov_arcmin = arcsec / 60

        if p["axesUnits"] == "code":
            size_kpc = p["sP"].units.arcsecToAngSizeKpcAtRedshift(arcsec)
            size = p["sP"].units.physicalKpcToCodeLength(size_kpc)
        elif p["axesUnits"] == "kpc":
            size = p["sP"].units.arcsecToAngSizeKpcAtRedshift(arcsec)
        elif p["axesUnits"] == "mpc":
            size = p["sP"].units.arcsecToAngSizeKpcAtRedshift(arcsec) / 1e3
        elif p["axesUnits"] == "arcsec":
            size = arcsec
        elif p["axesUnits"] == "arcmin":
            size = arcsec / 60
        elif p["axesUnits"] == "deg":
            size = arcsec / 60**2
        else:
            assert 0, "Unhandled."

        cen = [(pExtent[0] + pExtent[1]) / 2, (pExtent[2] + pExtent[3]) / 2]
        xmin = cen[0] - size / 2
        xmax = cen[0] + size / 2
        ymin = cen[1] - size / 2
        ymax = cen[1] + size / 2

        # draw with label
        color = "#ffffff"
        ax.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], "-", color=color)

        textOpts = {
            "color": color,
            "alpha": 1.0,
            "fontsize": 16,
            "horizontalalignment": "center",
            "verticalalignment": "bottom",
        }
        ax.text((xmin + xmax) / 2, ymax, "%d' FoV" % fov_arcmin, **textOpts)

    if "rVirFracs" in p and p["rVirFracs"]:
        # plot circles for N fractions of the virial radius
        xyPos = [p["boxCenter"][0], p["boxCenter"][1]]

        if p["relCoords"]:
            xyPos = [0.0, 0.0]

        if p["sP"].subhaloInd is not None and p["sP"].subhaloInd >= 0:
            # in the case that the box is not centered on the halo (e.g. offset quadrant), can use:
            sub = p["sP"].groupCatSingle(subhaloID=p["sP"].subhaloInd)

            if not p["relCoords"]:
                xyPos = sub["SubhaloPos"][p["axes"]]

        if p["axesUnits"] == "code":
            pass
        elif p["axesUnits"] == "kpc":
            xyPos = p["sP"].units.codeLengthToKpc(xyPos)
        elif p["axesUnits"] == "mpc":
            xyPos = p["sP"].units.codeLengthToMpc(xyPos)
        elif p["axesUnits"] in ["deg", "arcmin", "arcsec"]:
            assert p["relCoords"]  # makes the rest of this unneeded
            deg = p["axesUnits"] == "deg"
            amin = p["axesUnits"] == "arcmin"
            asec = p["axesUnits"] == "arcsec"
            xyPos = p["sP"].units.codeLengthToAngularSize(xyPos, deg=deg, arcmin=amin, arcsec=asec)
        else:
            raise Exception("Handle.")

        for rVirFrac in p["rVirFracs"]:
            rad = rVirFrac

            if p["fracsType"] == "rVirial":
                rad *= p["haloVirRad"]
            if p["fracsType"] == "rHalfMass":
                rad *= p["galHalfMass"]
            if p["fracsType"] == "rHalfMassStars":
                rad *= p["galHalfMassStars"]
                # if rad == 0.0:
                #    #print('Warning: Drawing frac [%.1f %s] is zero, use halfmass.' % (rVirFrac,p['fracsType']))
                #    #rad = rVirFrac * p['galHalfMass']
            if p["fracsType"] == "rhalf_stars_fof":
                # load custom stellar half mass radius (based on fof-scope), code length units
                rhalf_stars_fof = p["sP"].subhalos("rhalf_stars_fof_code")
                rad *= rhalf_stars_fof[p["sP"].subhaloInd]
            if p["fracsType"] == "codeUnits":
                rad *= 1.0
            if p["fracsType"] == "kpc":
                rad = p["sP"].units.physicalKpcToCodeLength(rad)

            if p["axesUnits"] == "code":
                pass
            elif p["axesUnits"] == "kpc":
                rad = p["sP"].units.codeLengthToKpc(rad)
            elif p["axesUnits"] == "mpc":
                rad = p["sP"].units.codeLengthToMpc(rad)
            elif p["axesUnits"] in ["deg", "arcmin", "arcsec"]:
                deg = p["axesUnits"] == "deg"
                amin = p["axesUnits"] == "arcmin"
                asec = p["axesUnits"] == "arcsec"
                rad = p["sP"].units.codeLengthToAngularSize(rad, deg=deg, arcmin=amin, arcsec=asec)
            else:
                raise Exception("Handle.")

            # show if circle is larger than 2 pixels
            pxScale = p["boxSizeImg"][p["axes"][0]] / p["nPixels"][0]

            color = "#ffffff"
            if rad > 5 * pxScale:
                c = plt.Circle((xyPos[0], xyPos[1]), rad, color=color, linewidth=1.5, fill=False, alpha=0.6)
                ax.add_artist(c)
            # else:
            #    print('Warning: Drawing radius at [%.1f %s] is zero or small, skip.' % (rVirFrac,p['fracsType']))

    if "labelZ" in p and p["labelZ"]:
        if p["sP"].redshift >= 0.99 or np.abs(np.round(10 * p["sP"].redshift) / 10 - p["sP"].redshift) < 1e-2:
            zStr = r"z$\,$=$\,$%.1f" % p["sP"].redshift
        else:
            zStr = r"z$\,$=$\,$%.2f" % p["sP"].redshift

        if p["sP"].redshift < 1e-3:
            zStr = r"z$\,$=$\,$0"

        if p["labelZ"] == "tage":
            zStr = "%5.2f billion years after the Big Bang" % p["sP"].units.redshiftToAgeFlat(p["sP"].redshift)

        xt = pExtent[1] - (pExtent[1] - pExtent[0]) * (0.01) * conf.nLinear  # upper right
        yt = pExtent[3] - (pExtent[3] - pExtent[2]) * (0.01) * conf.nLinear
        color = "white" if "textcolor" not in p else p["textcolor"]

        ax.text(
            xt, yt, zStr, color=color, alpha=1.0, size=conf.fontsize, ha="right", va="top"
        )  # same size as legend text

    if "labelScale" in p and p["labelScale"]:
        # add a scale bar: what is the total size of the plot (in code units?)
        extent = np.array(p["extent"])

        if p["axesUnits"] in ["deg", "arcmin", "arcsec"]:
            fac = 1 if p["axesUnits"] == "arcsec" else (60 if p["axesUnits"] == "arcmin" else 3600.0)
            extent = p["sP"].units.arcsecToAngSizeKpcAtRedshift(extent * fac)
            extent = p["sP"].units.physicalKpcToCodeLength(extent)
        if p["axesUnits"] == "kpc":
            extent = p["sP"].units.physicalKpcToCodeLength(extent)
        if p["axesUnits"] == "mpc":
            extent = p["sP"].units.physicalMpcToCodeLength(extent)

        scaleBarLen = (extent[1] - extent[0]) * 0.10  # 10% of plot width
        scaleBarLen /= p["sP"].HubbleParam  # ckpc/h -> ckpc (or cMpc/h -> cMpc)

        # if scale bar is more than (less than) 30% (20%) of width, reduce (increase)
        while scaleBarLen >= 0.3 * (extent[1] - extent[0]):
            scaleBarLen /= 1.4

        while scaleBarLen < 0.20 * (extent[1] - extent[0]):
            scaleBarLen *= 1.4

        # if scale bar is more than X Mpc/kpc, round to nearest X Mpc/kpc
        mpcFac = 1000.0 if p["sP"].mpcUnits else 1.0
        roundScales = np.array([10000.0, 1000.0, 1000.0, 100.0, 10.0, 1.0, 0.1]) / mpcFac

        for roundScale in roundScales:
            if scaleBarLen >= roundScale:
                scaleBarLen = roundScale * np.round(scaleBarLen / roundScale)

        scaleBarLen *= 1.5

        # actually plot size in code units (e.g. ckpc/h)
        scaleBarPlotLen = scaleBarLen * p["sP"].HubbleParam

        if p["labelScale"] == "physical":
            # convert size from comoving to physical
            scaleBarLen *= p["sP"].units.scalefac

            # want to round this display value
            for roundScale in roundScales:
                if scaleBarLen >= roundScale:
                    scaleBarLen = roundScale * np.round(scaleBarLen / roundScale)

            # make sure we draw the correct (rounded) length
            scaleBarPlotLen = p["sP"].units.physicalKpcToCodeLength(scaleBarLen * mpcFac)

        # label
        cmStr = "c" if (p["sP"].redshift > 0.0 and p["labelScale"] != "physical") else ""
        unitStrs = [cmStr + "pc", cmStr + "kpc", cmStr + "Mpc", cmStr + "Gpc"]  # comoving (drop 'c' if at z=0)
        unitInd = 1 if p["sP"].mpcUnits is False else 2

        scaleBarStr = "%d %s" % (scaleBarLen, unitStrs[unitInd])
        if scaleBarLen > 900:  # use Mpc label
            # scaleText = '%.2f' % (scaleBarLen/1000.0) if scaleBarLen/1000.0 < 10 else '%g' % (scaleBarLen/1000.0)
            scaleText = "%d" % (scaleBarLen / 1000.0) if scaleBarLen / 1000.0 < 10 else "%g" % (scaleBarLen / 1000.0)
            scaleBarStr = "%s %s" % (scaleText, unitStrs[unitInd + 1])
        if scaleBarLen < 1:  # use pc label
            scaleBarStr = "%g %s" % (scaleBarLen * 1000.0, unitStrs[unitInd - 1])

        if p["labelScale"] == "lightyears":
            scaleBarLen = scaleBarLen * 2.4 * p["sP"].units.scalefac * (p["sP"].units.kpc_in_ly / 1000)

            # want to round this display value
            for roundScale in roundScales:
                if scaleBarLen >= roundScale:
                    scaleBarLen = roundScale * np.round(scaleBarLen / roundScale)
                    scaleBarPlotLen = p["sP"].units.lightyearsToCodeLength(scaleBarLen * 1000)

            scaleBarStr = "%d thousand lightyears" % scaleBarLen

        lw = 2.5 * np.sqrt(conf.rasterPx[1] / 1000)
        y_off = np.clip(0.04 - 0.01 * 1000 / conf.rasterPx[1], 0.01, 0.02)
        yt_fac = np.clip(1.5 + 0.1 * 1000 / conf.rasterPx[1], 1.0, 2.0)

        x0 = extent[0] + (extent[1] - extent[0]) * (y_off * 720.0 / conf.rasterPx[0])  # upper left
        x1 = x0 + scaleBarPlotLen
        yy = extent[3] - (extent[3] - extent[2]) * (y_off * 720.0 / conf.rasterPx[1])
        yt = extent[3] - (extent[3] - extent[2]) * ((y_off * yt_fac) * 720.0 / conf.rasterPx[1])

        if p["axesUnits"] in ["deg", "arcmin", "arcsec"]:
            deg = p["axesUnits"] == "deg"
            amin = p["axesUnits"] == "arcmin"
            asec = p["axesUnits"] == "arcsec"
            x0 = p["sP"].units.codeLengthToAngularSize(x0, deg=deg, arcmin=amin, arcsec=asec)
            x1 = p["sP"].units.codeLengthToAngularSize(x1, deg=deg, arcmin=amin, arcsec=asec)
            yy = p["sP"].units.codeLengthToAngularSize(yy, deg=deg, arcmin=amin, arcsec=asec)
            yt = p["sP"].units.codeLengthToAngularSize(yt, deg=deg, arcmin=amin, arcsec=asec)
        if p["axesUnits"] == "kpc":
            x0 = p["sP"].units.codeLengthToKpc(x0)
            x1 = p["sP"].units.codeLengthToKpc(x1)
            yy = p["sP"].units.codeLengthToKpc(yy)
            yt = p["sP"].units.codeLengthToKpc(yt)
        if p["axesUnits"] == "mpc":
            x0 = p["sP"].units.codeLengthToMpc(x0)
            x1 = p["sP"].units.codeLengthToMpc(x1)
            yy = p["sP"].units.codeLengthToMpc(yy)
            yt = p["sP"].units.codeLengthToMpc(yt)

        color = "white" if "textcolor" not in p else p["textcolor"]

        ax.plot([x0, x1], [yy, yy], "-", color=color, lw=lw, alpha=1.0)
        ax.text(np.mean([x0, x1]), yt, scaleBarStr, color=color, alpha=1.0, size=conf.fontsize, ha="center", va="top")

    # text in a combined legend?
    legend_labels = []

    if "labelHalo" in p and p["labelHalo"] and p["sP"].subhaloInd >= 0:
        assert p["sP"].subhaloInd is not None

        subhalo = p["sP"].groupCatSingle(subhaloID=p["sP"].subhaloInd)
        halo = p["sP"].groupCatSingle(haloID=subhalo["SubhaloGrNr"])

        haloMass = p["sP"].units.codeMassToLogMsun(halo["Group_M_Crit200"])
        stellarMass = p["sP"].units.codeMassToLogMsun(subhalo["SubhaloMassInRadType"][p["sP"].ptNum("stars")])

        if "mhalo" in str(p["labelHalo"]):
            str1 = r"log M$_{\rm halo}$ = %.1f" % haloMass
            legend_labels.append(str1)
        if "mstar" in str(p["labelHalo"]):
            str2 = r"log M$_{\star}$ = %.1f" % stellarMass
            if np.isnan(stellarMass):
                str2 = r"log M$_{\star}$ = 0"
            legend_labels.append(str2)
        if "haloidorig" in str(p["labelHalo"]):
            assert p["sP"].simName == "TNG-Cluster"  # Halo ID from parent DMO box
            legend_labels.append("HaloID %d (#%d)" % (subhalo["SubhaloGrNr"], halo["GroupOrigHaloID"]))
        elif "haloid" in str(p["labelHalo"]):
            legend_labels.append("HaloID %d" % subhalo["SubhaloGrNr"])
        elif "id" in str(p["labelHalo"]):
            legend_labels.append("ID %d" % p["sP"].subhaloInd)
            # legend_labels.append( 'ID %d' % subhalo['SubhaloGrNr'] )

        if "sfr" in str(p["labelHalo"]):
            legend_labels.append(r"SFR = %.1f M$_\odot$ yr$^{-1}$" % subhalo["SubhaloSFRinRad"])
        if "redshift" in str(p["labelHalo"]):
            # legend_labels.append( 'z = %.1f, ID %d' % (p['sP'].redshift,p['sP'].subhaloInd))
            legend_labels.append("z = %.1f, ID %d" % (p["sP"].redshift, subhalo["SubhaloGrNr"]))

    if "labelCustom" in p and p["labelCustom"]:
        for label in p["labelCustom"]:
            legend_labels.append(label)

    if "labelAge" in p and p["labelAge"]:
        # age of the universe
        legend_labels.append("t = %.2f Gyr" % p["sP"].tage)

    if "labelSim" in p and p["labelSim"]:
        if isinstance(p["labelSim"], str):
            legend_labels.append(p["labelSim"])
        else:
            legend_labels.append(p["sP"].simName)

    # draw legend
    if len(legend_labels):
        # sort in order of string length, if first entry is longer than last
        if len(legend_labels[0]) > len(legend_labels[-1]):
            legend_labels.sort(key=lambda s: len(s))

        legend_lines = [plt.Line2D((0, 0), (0, 0), linestyle="") for _ in legend_labels]
        loc = p["legendLoc"] if "legendLoc" in p else "lower left"
        legend = ax.legend(
            legend_lines, legend_labels, fontsize=conf.fontsize, loc=loc, handlelength=0, handletextpad=0, borderpad=0
        )

        color = "white" if "textcolor" not in p else p["textcolor"]
        for text in legend.get_texts():
            text.set_color(color)


def addVectorFieldOverlay(p, conf, ax):
    """Add quiver or streamline overlay on top to visualization vector field data."""
    if "vecOverlay" not in p or not p["vecOverlay"]:
        return

    field_pt = None

    if p["vecOverlay"] == "bfield":
        assert p["rotMatrix"] is None  # otherwise need to handle like los-vel/velComps
        field_pt = "gas"
        field_name = "bfield"

    if "_vel" in p["vecOverlay"]:
        # we are handling rotation properly for the velocity field (e.g. 'gas_vel', 'stars_vel', 'dm_vel')
        field_pt, field_name = p["vecOverlay"].split("_")

    assert field_pt is not None

    field_x = field_name + "_" + ["x", "y", "z"][p["axes"][0]]
    field_y = field_name + "_" + ["x", "y", "z"][p["axes"][1]]
    nPixels = [40, 40] if "vecOverlaySizePx" not in p else p["vecOverlaySizePx"]
    qStride = 3  # for quiverplot, total number of ticks per axis is nPixels[i]/qStride
    # for streamlines, density=1 produces a 30x30 grid (linear scaling with density):
    density = [1.0, 1.0] if "vecOverlayDensity" not in p else p["vecOverlayDensity"]
    vecSliceWidth = 5.0 if "vecOverlayWidth" not in p else p["vecOverlayWidth"]  # pkpc
    arrowsize = 1.5  # for streamlines
    smoothFWHM = None

    # compress vector grids along third direction to more thin slice
    boxSizeImg = np.array(p["boxSizeImg"])
    boxSizeImg[3 - p["axes"][0] - p["axes"][1]] = p["sP"].units.physicalKpcToCodeLength(vecSliceWidth)

    # load two grids of vector length in plot-x and plot-y directions
    grid_x, _, _ = gridBox(
        p["sP"],
        p["method"],
        field_pt,
        field_x,
        nPixels,
        p["axes"],
        p["projType"],
        p["projParams"],
        p["boxCenter"],
        boxSizeImg,
        p["hsmlFac"],
        p["rotMatrix"],
        p["rotCenter"],
        p["remapRatio"],
        smoothFWHM=smoothFWHM,
    )

    grid_y, _, _ = gridBox(
        p["sP"],
        p["method"],
        field_pt,
        field_y,
        nPixels,
        p["axes"],
        p["projType"],
        p["projParams"],
        p["boxCenter"],
        boxSizeImg,
        p["hsmlFac"],
        p["rotMatrix"],
        p["rotCenter"],
        p["remapRatio"],
        smoothFWHM=smoothFWHM,
    )

    # load a grid of any quantity to use to color map the strokes
    grid_c, conf_c, _ = gridBox(
        p["sP"],
        p["method"],
        p["vecColorPT"],
        p["vecColorPF"],
        nPixels,
        p["axes"],
        p["projType"],
        p["projParams"],
        p["boxCenter"],
        boxSizeImg,
        p["hsmlFac"],
        p["rotMatrix"],
        p["rotCenter"],
        p["remapRatio"],
        smoothFWHM=smoothFWHM,
    )

    # create a unit vector at the position of each pixel
    grid_mag = np.sqrt(grid_x**2.0 + grid_y**2.0)

    w = np.where(grid_mag == 0.0)  # protect against zero magnitude
    grid_mag[w] = grid_mag.max() * 1e10  # set grid_x,y to zero in these cases

    grid_x /= grid_mag
    grid_y /= grid_mag

    # create arrow starting (tail) positions
    pxScale = p["boxSizeImg"][p["axes"]] / p["nPixels"]
    xx = np.linspace(p["extent"][0] + pxScale[0] / 2, p["extent"][1] - pxScale[0] / 2, nPixels[0])
    yy = np.linspace(p["extent"][2] + pxScale[1] / 2, p["extent"][3] - pxScale[1] / 2, nPixels[1])

    # prepare for streamline variable thickness
    maxSize = 4.0
    minSize = 0.5
    uniSize = 1.0

    grid_c2 = grid_c
    if p["vecOverlay"] == "bfield":
        # do a unit conversion such that we could actually make a quantitative streamplot (in progress)
        grid_c2 = 10.0**grid_c * 1e12  # [log G] -> [linear pG]

    grid_s = (maxSize - minSize) / (grid_c2.max() - grid_c2.min()) * (grid_c2 - grid_c2.min()) + minSize
    # grid_s /= 2

    # set normalization?
    norm = None
    if p["vecMinMax"] is not None:
        norm = mpl.colors.Normalize(vmin=p["vecMinMax"][0], vmax=p["vecMinMax"][1])

    # (A) plot white quivers
    if p["vecMethod"] == "A":
        assert norm is None
        ax.quiver(
            xx[::qStride],
            yy[::qStride],
            grid_x[::qStride, ::qStride],
            grid_y[::qStride, ::qStride],
            color="white",
            angles="xy",
            pivot="mid",
            headwidth=2.0,
            headlength=3.0,
        )

    # (B) plot colored quivers
    if p["vecMethod"] == "B":
        assert norm is None  # don't yet know how to handle
        ax.quiver(
            xx[::qStride],
            yy[::qStride],
            grid_x[::qStride, ::qStride],
            grid_y[::qStride, ::qStride],
            grid_c[::qStride, ::qStride],
            angles="xy",
            pivot="mid",
            width=0.0005,
            headwidth=0.0,
            headlength=0.0,
        )
        # legend for quiver length: (in progress) (q is return of ax.quiver())
        # ax.quiverkey(q, 1.1, 1.05, 10.0, 'label', labelpos='E', labelsep=0.1,  coordinates='figure')

    # (C) plot white streamlines, uniform thickness
    if p["vecMethod"] == "C":
        ax.streamplot(xx, yy, grid_x, grid_y, density=density, linewidth=None, arrowsize=arrowsize, color="white")

    # (D) plot white streamlines, thickness scaled by color quantity
    if p["vecMethod"] == "D":
        lw = 1.5 * grid_s
        c = ax.streamplot(xx, yy, grid_x, grid_y, density=density, linewidth=lw, arrowsize=arrowsize, color="white")
        c.lines.set_alpha(0.6)
        c.arrows.set_alpha(0.6)

    # (E) plot colored streamlines, uniform thickness
    if p["vecMethod"] == "E":
        c = ax.streamplot(
            xx,
            yy,
            grid_x,
            grid_y,
            density=density,
            linewidth=uniSize,
            color=grid_c,
            arrowsize=arrowsize,
            cmap="afmhot",
            norm=norm,
        )
        c.lines.set_alpha(0.6)
        c.arrows.set_alpha(0.6)

    # (F) plot colored streamlines, thickness also proportional to color quantity
    if p["vecMethod"] == "F":
        ax.streamplot(
            xx,
            yy,
            grid_x,
            grid_y,
            density=density,
            linewidth=grid_s,
            color=grid_c,
            arrowsize=arrowsize,
            cmap="afmhot",
            norm=norm,
        )


def addContourOverlay(p, conf, ax):
    """Add set of contours on top to visualize a second field."""
    if "contour" not in p or not p["contour"]:
        return

    field_pt, field_name = p["contour"]  # e.g. ['gas','vrad'] or ['stars','coldens_msunkpc2']

    nPixels = p["nPixels"] if "contourSizePx" not in p else p["contourSizePx"]

    # compress vector grids along third direction to more thin slice?
    boxSizeImg = np.array(p["boxSizeImg"])
    if "contourSliceDepth" in p:
        boxSizeImg[3 - p["axes"][0] - p["axes"][1]] = p["sP"].units.physicalKpcToCodeLength(p["contourSliceDepth"])

    # load grid of contour quantity
    if field_name == p["partField"]:
        # use current field
        grid_c = p["grid"]  # grid_data
    else:
        smoothFWHM = p["smoothFWHM"] if "smoothFWHM" in p else None
        hsmlFac = p["hsmlFac"] if p["partType"] == field_pt else defaultHsmlFac(field_pt)
        grid_c, conf_c, _ = gridBox(
            p["sP"],
            p["method"],
            field_pt,
            field_name,
            nPixels,
            p["axes"],
            p["projType"],
            p["projParams"],
            p["boxCenter"],
            boxSizeImg,
            hsmlFac,
            p["rotMatrix"],
            p["rotCenter"],
            p["remapRatio"],
            smoothFWHM=smoothFWHM,
        )

    if "contourSmooth" in p:  # px
        grid_c = gaussian_filter(grid_c, p["contourSmooth"], mode="reflect", truncate=5.0)

    # make pixel grid
    XX = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], grid_c.shape[0])
    YY = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], grid_c.shape[1])
    grid_x, grid_y = np.meshgrid(XX, YY)

    # contour options:
    #   'colors' can be a string e.g. 'white' or a list of strings/colors, one per level
    #   'alpha', 'cmap', 'linewidths' (num or list), 'linestyles' (num or list)
    contourOpts = {} if "contourOpts" not in p else p["contourOpts"]

    if "contourLevels" in p:
        # either [int] number of levels, or [list] of actual values
        ax.contour(grid_x, grid_y, grid_c, p["contourLevels"], **contourOpts)
    else:
        # automatic contour levels
        ax.contour(grid_x, grid_y, grid_c, **contourOpts)


def addCustomColorbars(
    fig, conf, config, heightFac, barAreaBottom, barAreaTop, color2, colWidth, leftNorm, hOffset=None
):
    """Add colorbar(s) with custom positioning and labeling, either below or above panels."""
    if not conf.colorbars:
        return

    factor = 0.80  # bar length, fraction of column width, 1.0=whole
    height = 0.025  # colorbar height, fraction of entire figure
    if hOffset is None:
        hOffset = 0.25  # padding between image and top of bar (fraction of bar height)
    tOffset = 0.15  # padding between top of bar and top of text label (fraction of bar height)
    lOffset = 0.02  # padding between colorbar edges and end label (frac of bar width)

    # factor = 0.65 # tng data release paper: tng_fields override
    # conf.fontsize = 13 # tng data release paper: tng_fields override
    # height = 0.047 # tng data release paper: tng_fields override

    height *= heightFac

    # if hasattr(conf, "fontsize"):  # and conf.nCols == 1:
    #    height *= (np.clip(conf.fontsize, 9, 32) / 9) ** (1 / 2)

    if barAreaTop == 0.0:
        # bottom
        bottomNormBar = barAreaBottom - height * (hOffset + 1.0)
        textTopY = -tOffset
        textMidY = 0.45  # pixel adjust down by 1 hack
    else:
        # top
        bottomNormBar = (1.0 - barAreaTop) + height * hOffset
        textTopY = 1.0 + tOffset
        textMidY = 0.45

    leftNormBar = leftNorm + 0.5 * colWidth * (1 - factor)
    posBar = [leftNormBar, bottomNormBar, colWidth * factor, height]

    # add bounding axis and draw colorbar
    cax = fig.add_axes(posBar)
    cax.set_axis_off()

    if "vecMinMax" in config:
        # norm = mpl.colors.Normalize(vmin=config['vecMinMax'][0], vmax=config['vecMinMax'][1])
        colorbar = mpl.colorbar.ColorbarBase(cax, cmap=config["ctName"], orientation="horizontal")
        valLimits = config["vecMinMax"]  # colorbar.get_clim()
    else:
        colorbar = plt.colorbar(cax=cax, orientation="horizontal")
        valLimits = plt.gci().get_clim()

    colorbar.outline.set_edgecolor(color2)

    # label, centered and below/above
    if hasattr(conf, "colorbarsmall") and conf.colorbarsmall:
        t = cax.text(
            0.5,
            textMidY,
            config["label"],
            color=color2,
            transform=cax.transAxes,
            size=conf.fontsize,
            ha="center",
            va="center",
        )
    else:
        t = cax.text(
            0.5,
            textTopY,
            config["label"],
            color=color2,
            transform=cax.transAxes,
            size=conf.fontsize,
            ha="center",
            va="top" if barAreaTop == 0.0 else "bottom",
        )

    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    if bb.y0 < 0:
        print("Not good, colorbar is falling off bottom of plot, need to increase barAreaHeight.")

    if "Stellar Composite" in config["label"]:
        return  # skip meaningless [0, ..., 255] labels

    if "colorbarnoticks" in config:
        return

    # tick labels, 5 evenly spaced inside bar
    # colorsA = [(1, 1, 1), (0.9, 0.9, 0.9), (0.8, 0.8, 0.8), (0.2, 0.2, 0.2), (0, 0, 0)]
    colorsB = ["white", "white", "white", "black", "black"]
    colorsB = ["white", "white", "white", "white", "white"]

    formatStr = "%.1f" if np.max(np.abs(valLimits)) < 100.0 else "%d"
    if np.max(np.abs(valLimits)) < 0.01:
        formatStr = "%.0e"

    cax.text(
        0.0 + lOffset,
        textMidY,
        formatStr % (1.0 * valLimits[0] + 0.0 * valLimits[1]),
        color=colorsB[0],
        size=conf.fontsize,
        ha="left",
        va="center",
        transform=cax.transAxes,
    )
    if not hasattr(conf, "colorbarsmall") or not conf.colorbarsmall:
        cax.text(
            0.25,
            textMidY,
            formatStr % (0.75 * valLimits[0] + 0.25 * valLimits[1]),
            color=colorsB[1],
            size=conf.fontsize,
            ha="center",
            va="center",
            transform=cax.transAxes,
        )
        cax.text(
            0.5,
            textMidY,
            formatStr % (0.5 * valLimits[0] + 0.5 * valLimits[1]),
            color=colorsB[2],
            size=conf.fontsize,
            ha="center",
            va="center",
            transform=cax.transAxes,
        )
        cax.text(
            0.75,
            textMidY,
            formatStr % (0.25 * valLimits[0] + 0.75 * valLimits[1]),
            color=colorsB[3],
            size=conf.fontsize,
            ha="center",
            va="center",
            transform=cax.transAxes,
        )
    cax.text(
        1.0 - lOffset,
        textMidY,
        formatStr % (0.0 * valLimits[0] + 1.0 * valLimits[1]),
        color=colorsB[4],
        size=conf.fontsize,
        ha="right",
        va="center",
        transform=cax.transAxes,
    )


def _getPlotExtent(extent, axesUnits, projType, sP):
    """Helper function, manipulate input extent given requested axesUnits."""
    if axesUnits == "code":
        pExtent = extent
    if axesUnits == "kpc":
        pExtent = sP.units.codeLengthToKpc(extent)
    if axesUnits == "mpc":
        pExtent = sP.units.codeLengthToMpc(extent)
    if axesUnits == "arcsec":
        pExtent = sP.units.codeLengthToAngularSize(extent, arcsec=True)
    if axesUnits == "arcmin":
        pExtent = sP.units.codeLengthToAngularSize(extent, arcmin=True)
    if axesUnits == "deg":
        if sP.redshift == 0.0:
            sP.redshift = 0.1  # temporary
        pExtent = sP.units.codeLengthToAngularSize(extent, deg=True)
        # shift to arbitrary center at (0,0)
        if pExtent[0] == 0.0 and pExtent[2] == 0.0:
            assert pExtent[1] == pExtent[3]  # box, not halo, imaging
            pExtent -= pExtent[1] / 2
    if projType == "equirectangular":
        assert axesUnits == "rad_pi"
        pExtent = [0, 2, 0, 1]  # in units of pi
    if projType == "azimuthalequidistant":
        assert axesUnits == "rad_pi"
        pExtent = [0, 1, 0, 1]  # in units of pi
    if projType == "mollweide":
        assert axesUnits == "rad_pi"
        pExtent = [0, 2, 0, 1]  # in units of pi

    return pExtent


def renderMultiPanel(panels, conf):
    """Generalized plotting function which produces a multi-panel plot, all of which can vary in their configuration.

    Args:
      conf (dict): Global plot configuration options. See :py:func:`~vis.halo.renderSingleHalo` and
        :py:func:`~vis.box.renderBox` for available options and their default values.

      panels (list): Each panel must be a dictionary containing the following keys. See
        :py:func:`~vis.halo.renderSingleHalo` and :py:func:`~vis.box.renderBox` for available options
        and their default values.

    Returns:
      None. Figure is produced in current directory.
    """
    assert conf.plotStyle in ["open", "open_black", "edged", "edged_black"]
    assert len(panels) > 0

    color1 = "black" if "_black" in conf.plotStyle else "white"
    color2 = "white" if "_black" in conf.plotStyle else "black"

    if hasattr(conf, "sizePath"):
        makedirs(conf.savePath, exist_ok=True)

    # plot sizing and arrangement
    sizeFac = np.array(conf.rasterPx) / mpl.rcParams["savefig.dpi"]
    nRows = int(np.floor(np.sqrt(len(panels)))) if not hasattr(conf, "nRows") else conf.nRows
    nCols = int(np.ceil(len(panels) / nRows)) if not hasattr(conf, "nCols") else conf.nCols
    aspect = nRows / nCols

    conf.nCols = nCols
    conf.nRows = nRows

    # approximate font-size invariance with changing rasterPx
    conf.nLinear = conf.nCols if conf.nCols > conf.nRows else conf.nRows
    min_fontsize = 14 if "edged" in conf.plotStyle else 12
    max_fontsize = 60 if conf.nLinear <= 2 else 40
    fontsize_exp = 0.8
    if not hasattr(conf, "fontsize"):
        conf.fontsize = (conf.rasterPx[0] / 1000) ** fontsize_exp * 10 * (conf.nCols**fontsize_exp) * 1.2
        conf.fontsize = int(np.clip(conf.fontsize, min_fontsize, max_fontsize))

    if conf.plotStyle in ["open", "open_black"]:
        # start plot
        fig = plt.figure(facecolor=color1)

        widthFacCBs = 1.167 if conf.colorbars else 1.0
        size_x = sizeFac[0] * nRows * widthFacCBs / aspect
        if panels[0]["remapRatio"] is not None:  # rough correction for single-panel remapped images
            size_x *= panels[0]["remapRatio"][0] / panels[0]["remapRatio"][1]
        size_y = sizeFac[1] * nRows
        size_x = int(np.round(size_x * 100.0))  # npixels
        size_y = int(np.round(size_y * 100.0))
        if size_x % 2 == 1:
            size_x += 1  # must be even for yuv420p pixel format x264 encode
        if size_y % 2 == 1:
            size_y += 1
        size_x /= 100.0
        size_y /= 100.0
        fig.set_size_inches(size_x, size_y)

        # for each panel: paths and render setup
        for i, p in enumerate(panels):
            if p["boxSizeImg"] is None:
                continue  # blank panel

            # grid projection for image
            grid, config, _ = gridBox(**p)

            assert "splitphase" not in p
            if "grid" in p:
                print("NOTE: Overriding computed image grid with input grid!")
                grid = p["grid"]
            else:
                p["grid_data"] = grid  # attach for later use

            if "colorbarlabel" in p:
                config["label"] = p["colorbarlabel"]
                print("NOTE: Overriding colorbar label with input label.")

            if "mock_redshift" in p:  # collab.rubin.hubbleMCT_gibleVis()
                old_redshift = p["sP"].redshift
                p["sP"].setRedshift(p["mock_redshift"])
                print(f"Pretending snapshot is at z={p['mock_redshift']:.2f} instead of z={old_redshift:.2f}.")

            sP = p["sP"]

            # create this panel, and label axes and title
            ax = fig.add_subplot(nRows, nCols, i + 1)

            if conf.title:
                idStr = " (id=" + str(sP.subhaloInd) + ")" if not sP.isZoom and sP.subhaloInd is not None else ""

                if np.isfinite(sP.redshift):
                    # cosmological
                    ax.set_title("%s z=%d%s" % (sP.simName, sP.redshift, idStr))
                    if sP.redshift != int(sP.redshift):
                        ax.set_title("%s z=%3.1f%s" % (sP.simName, sP.redshift, idStr))
                    if sP.redshift / 0.1 != int(sP.redshift / 0.1):
                        ax.set_title("%s z=%4.2f%s" % (sP.simName, sP.redshift, idStr))
                elif np.isfinite(sP.time):
                    # non-cosmological
                    ax.set_title(sP.simName)
                    # ax.set_title('%s t=%6.3f' % (sP.simName,sP.time))

            if "title" in p and p["title"] is not None:
                ax.set_title(p["title"])

            axStrs = {
                "code": "[ ckpc/h ]",
                "kpc": "[ pkpc ]",
                "mpc": "[ Mpc ]",
                "arcsec": "[ arcsec ]",
                "arcmin": "[ arcmin ]",
                "deg": "[ degrees ]",
                "rad_pi": r" [ radians / $\pi$ ]",
            }
            if p["sP"].mpcUnits:
                axStrs["code"] = "[ cMpc/h ]"
            axStr = axStrs[p["axesUnits"]]
            ax.set_xlabel(["x", "y", "z"][p["axes"][0]] + " " + axStr)
            ax.set_ylabel(["x", "y", "z"][p["axes"][1]] + " " + axStr)
            if p["axesUnits"] in ["arcsec", "arcmin", "deg"]:
                ax.set_xlabel(r"$\alpha$ " + axStr)  # e.g. right ascension
                ax.set_ylabel(r"$\delta$ " + axStr)  # e.g. declination
            if p["axesUnits"] in ["rad_pi"]:
                ax.set_xlabel(r"$\theta$ " + axStr)  # e.g. longitude
                ax.set_ylabel(r"$\phi$ " + axStr)  # e.g. latitude

            setAxisColors(ax, color2)

            # rotation? indicate transformation with axis labels
            if p["rotMatrix"] is not None:
                old_1 = np.zeros(3, dtype="float32")
                old_2 = np.zeros(3, dtype="float32")
                old_1[p["axes"][0]] = 1.0
                old_2[p["axes"][1]] = 1.0

                # new_1 = np.transpose(np.dot(p["rotMatrix"], old_1))
                # new_2 = np.transpose(np.dot(p["rotMatrix"], old_2))

                # ax.set_xlabel( 'rotated: %4.2fx %4.2fy %4.2fz %s' % (new_1[0], new_1[1], new_1[2], axStr))
                # ax.set_ylabel( 'rotated: %4.2fx %4.2fy %4.2fz %s' % (new_2[0], new_2[1], new_2[2], axStr))
                ax.set_xlabel("x %s" % axStr)
                ax.set_ylabel("y %s" % axStr)

            # color mapping (handle defaults and overrides)
            vMM = p["valMinMax"] if p["valMinMax"] is not None else config["vMM_guess"]
            plaw = p["plawScale"] if "plawScale" in p else None
            if "plawScale" in config:
                plaw = config["plawScale"]
            if "plawScale" in p:
                plaw = p["plawScale"]
            cenVal = p["cmapCenVal"] if "cmapCenVal" in p else None
            if "cmapCenVal" in config:
                cenVal = config["cmapCenVal"]
            ctName = p["ctName"] if p["ctName"] is not None else config["ctName"]
            numColors = p["numColors"] if "numColors" in p else None

            cmap = loadColorTable(ctName, valMinMax=vMM, plawScale=plaw, cmapCenterVal=cenVal, numColors=numColors)

            cmap.set_bad(color="#000000", alpha=1.0)  # use black for nan pixels
            grid = np.ma.array(grid, mask=np.isnan(grid))

            # place image
            pExtent = _getPlotExtent(p["extent"], p["axesUnits"], p["projType"], p["sP"])

            plt.imshow(grid, extent=pExtent, cmap=cmap, aspect=grid.shape[0] / grid.shape[1])

            ax.autoscale(False)
            if cmap is not None:
                plt.clim(vMM)

            addBoxMarkers(p, conf, ax, pExtent)

            addVectorFieldOverlay(p, conf, ax)

            addContourOverlay(p, conf, ax)

            # colorbar
            if conf.colorbars:
                pad = np.clip(conf.rasterPx[0] / 6000.0, 0.05, 0.4)  # 0.2 for 1200px
                cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=pad)
                setAxisColors(cax, color2)

                cb = plt.colorbar(cax=cax)
                cb.outline.set_edgecolor(color2)
                cb.ax.set_ylabel(config["label"])

                if "colorbarnoticks" in p:
                    cb.set_ticks([])

            padding = conf.rasterPx[0] / 240.0
            ax.tick_params(axis="x", which="major", labelsize=conf.fontsize)
            ax.tick_params(axis="y", which="major", labelsize=conf.fontsize)
            ax.xaxis.label.set_size(conf.fontsize)
            ax.yaxis.label.set_size(conf.fontsize)
            ax.title.set_fontsize(conf.fontsize - 2)
            ax.tick_params(axis="both", which="major", pad=padding)

            if conf.colorbars:
                cb.ax.tick_params(axis="y", which="major", labelsize=conf.fontsize)
                cb.ax.yaxis.label.set_size(conf.fontsize)
                cb.ax.tick_params(axis="both", which="major", pad=padding)

            # post-render function hook?
            if "f_post" in p and callable(p["f_post"]):
                p["f_post"](ax, p, conf)

            if "mock_redshift" in p:
                p["sP"].setRedshift(old_redshift)

        # if nRows == 1 and nCols == 3:
        #    plt.subplots_adjust(top=0.97, bottom=0.06)  # fix degenerate case

    if conf.plotStyle in ["edged", "edged_black"]:
        # colorbar plot area sizing
        aspect = float(conf.rasterPx[1]) / conf.rasterPx[0] if hasattr(conf, "rasterPx") else 1.0
        barAreaHeight = 0.007 * conf.fontsize**fontsize_exp / aspect
        barAreaHeight = np.clip(barAreaHeight, 0.02, 0.12)

        if not conf.colorbars:
            barAreaHeight = 0.0

        # check uniqueness of panel (partType,partField,valMinMax)'s
        pPartTypes = set()
        pPartFields = set()
        pValMinMaxes = set()

        for p in panels:
            pPartTypes.add(p["partType"])
            pPartFields.add(p["partField"].replace("_dustdeplete", ""))
            pValMinMaxes.add(str(p["valMinMax"]))

        # if all panels in the entire figure are the same, we will do 1 single colorbar
        oneGlobalColorbar = False

        if len(pPartTypes) == 1 and len(pPartFields) == 1 and len(pValMinMaxes) == 1:
            oneGlobalColorbar = True

        # check if we have multiple panels with auto min-max scalings, if so, avoid single (wrong) colorbar
        if oneGlobalColorbar:
            field_counts = {}
            for panel in panels:
                key = "%s-%s" % (panel["partType"], panel["partField"])
                if key not in field_counts:
                    field_counts[key] = 0
                field_counts[key] += panel["valMinMax"] is None  # auto

            for k, v in field_counts.items():
                if v > 1 and "stellarComp" not in k:
                    print(f"WARNING: [{k}] has multiple panels with auto cbar minmax. Disabling global colorbar.")
                    oneGlobalColorbar = False

        # height of colorbar is this value times a constant (as a fraction of the figure size)
        heightFac = 1.0 * conf.nLinear**0.4

        if nRows == 2 and not oneGlobalColorbar:
            # two rows, special case, colors on top and bottom, every panel can be different
            barAreaTop = 0.5 * barAreaHeight
            barAreaBottom = 0.5 * barAreaHeight
            heightFac *= 0.5
        else:
            # colorbars on the bottom of the plot, one per column (columns should be same field/valMinMax)
            barAreaTop = 0.0
            barAreaBottom = barAreaHeight

        if nRows == 2 and oneGlobalColorbar:
            barAreaBottom *= 0.7
            heightFac *= 0.7

        # colorbar has its own space, or is on top of the plot?
        barTop = barAreaTop  # used to draw bars
        barBottom = barAreaBottom  # used to draw bars

        if conf.colorbarOverlay:
            # used to resize actual panels, so set to zero
            barAreaTop = 0.0
            barAreaBottom = 0.0

        # variable-height rows? e.g. face-on and edge-on views together
        varRowHeights = False
        nShortPanels = 0

        if len(panels) > 1:
            for p in panels:
                if p["nPixels"][1] <= 0.5 * p["nPixels"][0] and "rotation" in p and p["rotation"] == "edge-on":
                    varRowHeights = True
                    rowHeightRatio = p["nPixels"][1] / p["nPixels"][0]  # e.g. 0.25 for 4x longer than tall
                    nShortPanels += 1

        # if varRowHeights and nRows == 2 and nCols == 1: # single face-on edge-on combination
        #    barAreaBottom *= (1-rowHeightRatio/2)
        if varRowHeights and nRows == 2:
            barAreaTop = 0.0  # disable top set of bars

        assert nShortPanels / nCols == np.round(nShortPanels / nCols)  # exact number of panels to make full rows
        nShortRows = nShortPanels / nCols

        # start plot
        fig = plt.figure(layout="none", facecolor=color1)

        width_in = sizeFac[0] * np.ceil(nCols)
        height_in = sizeFac[1] * np.ceil(nRows)

        if varRowHeights:
            barAreaBottom /= np.sqrt(rowHeightRatio)
            assert nShortRows == nRows / 2  # otherwise unexpected configuration

            nTallRows = nRows - nShortRows  # == nRows/2
            rowHeightTall = (1.0 - barAreaTop - barAreaBottom) * (1.0 / (1 + rowHeightRatio)) / nTallRows
            rowHeightShort = (1.0 - barAreaTop - barAreaBottom) * (rowHeightRatio / (1 + rowHeightRatio)) / nShortRows

            height_in = sizeFac[1] * nTallRows + sizeFac[1] * nShortRows * rowHeightRatio

        rowHeight = (1.0 - barAreaTop - barAreaBottom) / np.ceil(nRows)
        height_in *= 1 / (1.0 - barAreaTop - barAreaBottom)  # account for colorbar areas

        # make sure pixel number in both width and height is even
        width_in = np.round(width_in * mpl.rcParams["savefig.dpi"] / 2) * 2 / mpl.rcParams["savefig.dpi"]
        height_in = np.round(height_in * mpl.rcParams["savefig.dpi"] / 2) * 2 / mpl.rcParams["savefig.dpi"]
        fig.set_size_inches(width_in, height_in)

        # for each panel: paths and render setup
        for i, p in enumerate(panels):
            if p["boxSizeImg"] is None:
                continue  # blank panel
            # grid projection for image
            grid, config, _ = gridBox(**p)

            if "grid" in p:
                print("NOTE: Overriding computed image grid with input grid!")
                grid = p["grid"]
            else:
                p["grid_data"] = grid  # attach for later use

            if "colorbarnoticks" in p:
                config["colorbarnoticks"] = p["colorbarnoticks"]

            # render tweaks
            if "splitphase" in p:
                print("NOTE: Rendering fraction of grid, phase = %s!" % p["splitphase"])
                splitPart, totParts = p["splitphase"]
                splitRange = pSplitRange([0, grid.shape[1]], totParts, splitPart)
                grid = grid[:, splitRange[0] : splitRange[1]]
                fig.set_size_inches(width_in / totParts, height_in)

            if "mock_redshift" in p:  # collab.rubin.hubbleMCT_gibleVis()
                old_redshift = sP.redshift
                p["sP"].setRedshift(p["mock_redshift"])
                print(f"Pretending snapshot is at z={p['mock_redshift']:.2f} instead of z={old_redshift:.2f}.")

            # set axes coordinates and add
            curRow = np.floor(i / nCols)
            curCol = i % nCols

            colWidth = 1.0 / np.ceil(nCols)
            leftNorm = colWidth * curCol
            bottomNorm = (1.0 - barAreaTop) - rowHeight * (curRow + 1)

            if varRowHeights:
                curRowTall = np.floor(curRow / 2.0)  # note: hard-coded 'alternating' logic here
                curRowShort = np.floor((curRow + 1) / 2.0)

                if p["nPixels"][1] <= 0.5 * p["nPixels"][0]:
                    # short/edge-on row
                    rowHeight = rowHeightShort
                else:
                    # tall/face-on row
                    rowHeight = rowHeightTall

                bottomNorm = (1.0 - barAreaTop) - rowHeightTall * (curRowTall + 1) - rowHeightShort * (curRowShort)

            pos = [leftNorm, bottomNorm, colWidth, rowHeight]

            ax = fig.add_axes(pos, facecolor=color1)
            ax.set_axis_off()
            setAxisColors(ax, color2)

            # color mapping (handle defaults and overrides)
            vMM = p["valMinMax"] if p["valMinMax"] is not None else config["vMM_guess"]
            plaw = p["plawScale"] if "plawScale" in p else None
            if "plawScale" in config:
                plaw = config["plawScale"]
            if "plawScale" in p:
                plaw = p["plawScale"]
            cenVal = p["cmapCenVal"] if "cmapCenVal" in p else None
            if "cmapCenVal" in config:
                cenVal = config["cmapCenVal"]
            ctName = p["ctName"] if p["ctName"] is not None else config["ctName"]
            numColors = p["numColors"] if "numColors" in p else None

            cmap = loadColorTable(ctName, valMinMax=vMM, plawScale=plaw, cmapCenterVal=cenVal, numColors=numColors)

            # DEBUG: dump raw 16-bit tiff image
            if 0:
                import skimage.io

                norm = mpl.colors.Normalize(vmin=vMM[0], vmax=vMM[1])
                mVal = np.uint16(65535)
                grid_out = np.round(cmap(norm(grid))[:, :, :3] * mVal).astype("uint16")
                grid_out = grid_out[::-1, :, :]  # np.transpose(grid_out, axes=[1,0,2])
                skimage.io.imsave(conf.saveFilename.replace(".png", ".tif"), grid_out, plugin="tifffile")

            # use black for nan pixels
            cmap.set_bad(color="#000000", alpha=1.0)
            if p["projType"] == "mollweide" and "_black" not in conf.plotStyle:
                # use white around mollweide edges
                cmap.set_bad(color="#ffffff", alpha=1.0)
                if "textcolor" not in p or p["textcolor"] in ["white", "#fff", "#ffffff"]:
                    p["textcolor"] = "black"

            grid = np.ma.array(grid, mask=np.isnan(grid))

            # place image
            pExtent = _getPlotExtent(p["extent"], p["axesUnits"], p["projType"], p["sP"])

            if "splitphase" in p:
                pExtent[1] /= totParts

            plt.imshow(grid, extent=pExtent, cmap=cmap, aspect="equal")  # float(grid.shape[0])/grid.shape[1]
            ax.autoscale(False)  # disable re-scaling of axes with any subsequent ax.plot()
            if cmap is not None:
                plt.clim(vMM)

            addBoxMarkers(p, conf, ax, pExtent)

            addVectorFieldOverlay(p, conf, ax)

            addContourOverlay(p, conf, ax)

            # post-render function hook?
            if "f_post" in p and callable(p["f_post"]):
                p["f_post"](ax, p, conf)

            # colobar(s)
            if oneGlobalColorbar:
                continue

            if nRows == 2:
                # both above and below, one per column
                # if conf.colorbarOverlay:
                #    heightFac *= 0.7

                if curRow == 0 and (barAreaTop > 0 or conf.colorbarOverlay):
                    addCustomColorbars(fig, conf, config, heightFac, 0.0, barTop, color2, colWidth, leftNorm)

                if curRow == nRows - 1 and (barAreaBottom > 0 or conf.colorbarOverlay):
                    addCustomColorbars(fig, conf, config, heightFac, barBottom, 0.0, color2, colWidth, leftNorm)

            if nRows == 1 or (nRows > 2 and curRow == nRows - 1):
                # only below, one per column
                addCustomColorbars(fig, conf, config, heightFac, barBottom, barTop, color2, colWidth, leftNorm)

            if "vecColorbar" in p and p["vecColorbar"] and not oneGlobalColorbar:
                raise Exception("Only support vecColorbar addition with oneGlobalColorbar type configuration.")

            if "mock_redshift" in p:
                p["sP"].setRedshift(old_redshift)

        # one global colorbar? centered at bottom
        if oneGlobalColorbar:
            widthFrac = 0.8
            hOffset = None

            if "vecColorbar" not in p or not p["vecColorbar"]:
                # normal
                leftNorm = 0.5 - widthFrac / 2
                addCustomColorbars(
                    fig, conf, config, heightFac, barBottom, barTop, color2, widthFrac, leftNorm, hOffset
                )
            else:
                # normal, offset to the left
                addCustomColorbars(fig, conf, config, heightFac, barBottom, barTop, color2, widthFrac, 0.05)

                # colorbar for the vector field visualization, offset to the right
                _, vConfig, _ = gridOutputProcess(
                    p["sP"], np.zeros(2), p["vecColorPT"], p["vecColorPF"], [1, 1], "ortho", 1.0
                )
                vConfig["vecMinMax"] = p["vecMinMax"]
                vConfig["ctName"] = p["vecColormap"]

                addCustomColorbars(fig, conf, vConfig, heightFac, barBottom, barTop, color2, widthFrac, 0.55)

    # note: conf.saveFilename may be an in-memory buffer, or an actual filesystem path
    fig.savefig(conf.saveFilename, format=conf.outputFmt, facecolor=fig.get_facecolor(), bbox_inches=0)
    plt.close(fig)
    plt.rcParams.update({"figure.autolayout": True})
