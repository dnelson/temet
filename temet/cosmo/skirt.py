"""
Prepare simulation outputs for SKIRT9 radiative transfer runs, execute, and vis/analyze.
"""

import subprocess
import time
from os.path import isfile

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..util.helper import logZeroMin, logZeroNaN
from ..util.rotation import momentOfInertiaTensor, rotateCoordinateArray, rotationMatricesFromInertiaTensor
from ..util.simParams import simParams


stars_source_fsps = """
    <ParticleSource filename="{filenameStars}" importVelocity="true" importVelocityDispersion="false" useColumns=""
    sourceWeight="1" wavelengthBias="0.5">
        <sedFamily type="SEDFamily">
            <FSPSSEDFamily imf="Chabrier"/>
        </sedFamily>
        <wavelengthBiasDistribution type="WavelengthDistribution">
            <LogWavelengthDistribution minWavelength="0.01 micron" maxWavelength="1e3 micron"/>
            <!--<LinWavelengthDistribution minWavelength="0.09 micron" maxWavelength="1.0 micron"/>-->
        </wavelengthBiasDistribution>
    </ParticleSource>
"""

stars_source_bc03mappings3 = """
    <!-- young -->
    <ParticleSource filename="{filenameStars1}" importVelocity="true" importVelocityDispersion="false" useColumns=""
    sourceWeight="1" wavelengthBias="0.5">
        <smoothingKernel type="SmoothingKernel">
            <CubicSplineSmoothingKernel/>
        </smoothingKernel>
        <sedFamily type="SEDFamily">
            <MappingsSEDFamily/>
        </sedFamily>
        <wavelengthBiasDistribution type="WavelengthDistribution">
            <LogWavelengthDistribution minWavelength="0.01 micron" maxWavelength="1e3 micron"/>
            <!--<LinWavelengthDistribution minWavelength="0.09 micron" maxWavelength="1.0 micron"/>-->
        </wavelengthBiasDistribution>
    </ParticleSource>

    <!-- old -->
    <ParticleSource filename="{filenameStars2}" importVelocity="true" importVelocityDispersion="false" useColumns=""
    sourceWeight="1" wavelengthBias="0.5">
        <smoothingKernel type="SmoothingKernel">
            <CubicSplineSmoothingKernel/>
        </smoothingKernel>
        <sedFamily type="SEDFamily">
            <BruzualCharlotSEDFamily imf="Chabrier" resolution="High"/>
        </sedFamily>
        <wavelengthBiasDistribution type="WavelengthDistribution">
            <LogWavelengthDistribution minWavelength="0.01 micron" maxWavelength="1e3 micron"/>
            <!--<LinWavelengthDistribution minWavelength="0.09 micron" maxWavelength="1.0 micron"/>-->
        </wavelengthBiasDistribution>
    </ParticleSource>
"""

dust_extinction = """
    <!-- only included if simMode is ExtinctionOnly -->
    <extinctionOnlyOptions type="ExtinctionOnlyOptions">
        <ExtinctionOnlyOptions storeRadiationField="false"/>
    </extinctionOnlyOptions>
"""

dust_emission = """
    <!-- only included if simMode is DustEmissionWithSelfAbsorption -->
    <dustEmissionOptions type="DustEmissionOptions">
        <!-- NOTE: Equilibrium can change to 'Stochastic' for non-LTE -->
        <DustEmissionOptions dustEmissionType="Equilibrium" storeEmissionRadiationField="false"
        secondaryPacketsMultiplier="1" spatialBias="0.5" wavelengthBias="0.5">
            <radiationFieldWLG type="DisjointWavelengthGrid">
                <LogWavelengthGrid minWavelength="{minWavelength}" maxWavelength="{maxWavelength}"
                numWavelengths="{numWavelengths}"/>
            </radiationFieldWLG>
            <dustEmissionWLG type="DisjointWavelengthGrid">
                <LogWavelengthGrid minWavelength="{minWavelength}" maxWavelength="{maxWavelength}"
                numWavelengths="{numWavelengths}"/>
            </dustEmissionWLG>
            <wavelengthBiasDistribution type="WavelengthDistribution">
                <LogWavelengthDistribution minWavelength="0.01 micron" maxWavelength="1e3 micron"/>
                <!--<LinWavelengthDistribution minWavelength="0.09 micron" maxWavelength="1.0 micron"/>-->
            </wavelengthBiasDistribution>
        </DustEmissionOptions>
    </dustEmissionOptions>
"""

dust_selfabs = """
    <dustSelfAbsorptionOptions type="DustSelfAbsorptionOptions">
        <DustSelfAbsorptionOptions minIterations="1" maxIterations="10" maxFractionOfPrimary="0.01"
        maxFractionOfPrevious="0.03" iterationPacketsMultiplier="1"/>
    </dustSelfAbsorptionOptions>
"""

ski_template = """
<?xml version="1.0" encoding="UTF-8"?>
<skirt-simulation-hierarchy type="MonteCarloSimulation" format="9" producer="SKIRT v9.0 (git 9691c88)" time="">
    <MonteCarloSimulation simulationMode="{simMode}" numPackets="{numPackets}">

        <random type="Random">
            <Random seed="424242"/>
        </random>

        <units type="Units">
            <ExtragalacticUnits fluxOutputStyle="Neutral"/>
        </units>

        <sourceSystem type="SourceSystem">
            <SourceSystem minWavelength="{minWavelength}" maxWavelength="{maxWavelength}" sourceBias="0.5">
                <sources type="Source">
{starsSourceModel}
                </sources>
            </SourceSystem>
        </sourceSystem>

        <mediumSystem type="MediumSystem">
            <MediumSystem>
                <photonPacketOptions type="PhotonPacketOptions">
                    <PhotonPacketOptions minWeightReduction="1e4" minScattEvents="0" pathLengthBias="0.5"/>
                </photonPacketOptions>

{dustModelParameters}

                <media type="Medium">
                    <VoronoiMeshMedium filename="{filenameGas}" minX="{minX}" maxX="{maxX}" minY="{minY}" maxY="{maxY}"
                    minZ="{minZ}" maxZ="{maxZ}" massType="MassDensity" massFraction="1" importMetallicity="false"
                    importTemperature="false" importVelocity="true" importMagneticField="false"
                    importVariableMixParams="false" useColumns="">
                        <materialMix type="MaterialMix">
                            <!--<MeanInterstellarDustMix/>--> <!-- faster for extinction only -->
                            <ZubkoDustMix numSilicateSizes="15" numGraphiteSizes="15" numPAHSizes="10"/>
                        </materialMix>
                    </VoronoiMeshMedium>
                </media>

                <grid type="SpatialGrid">
                    <VoronoiMeshSpatialGrid minX="{minX}" maxX="{maxX}" minY="{minY}" maxY="{maxY}"
                    minZ="{minZ}" maxZ="{maxZ}" policy="ImportedMesh">
                    </VoronoiMeshSpatialGrid>
                </grid>

            </MediumSystem>
        </mediumSystem>

        <instrumentSystem type="InstrumentSystem">
            <InstrumentSystem>
                <defaultWavelengthGrid type="WavelengthGrid">
                    <LogWavelengthGrid minWavelength="{minWavelength}" maxWavelength="{maxWavelength}"
                    numWavelengths="{numWavelengths}"/>
                </defaultWavelengthGrid>
                <instruments type="Instrument">
                    <!-- recordStatistics can be false to save space, it gives debug info on the photons -->
                    <FullInstrument instrumentName="faceon" distance="10 Mpc"
                    inclination="0 deg" azimuth="0 deg" roll="0 deg"
                    fieldOfViewX="{fovX}" numPixelsX="{numPixelsX}" centerX="0 pc"
                    fieldOfViewY="{fovY}" numPixelsY="{numPixelsY}" centerY="0 pc"
                    recordComponents="true" numScatteringLevels="0" recordPolarization="false" recordStatistics="true"/>
                    <FullInstrument instrumentName="edgeon" distance="10 Mpc"
                    inclination="90 deg" azimuth="-90 deg" roll="90 deg"
                    fieldOfViewX="{fovX}" numPixelsX="{numPixelsX}" centerX="0 pc"
                    fieldOfViewY="{fovY}" numPixelsY="{numPixelsY}" centerY="0 pc"
                    recordComponents="true" numScatteringLevels="0" recordPolarization="false" recordStatistics="true"/>
                    <FullInstrument instrumentName="random" distance="10 Mpc"
                    inclination="{incRandom}" azimuth="{aziRandom}" roll="0 deg"
                    fieldOfViewX="{fovX}" numPixelsX="{numPixelsX}" centerX="0 pc"
                    fieldOfViewY="{fovY}" numPixelsY="{numPixelsY}" centerY="0 pc"
                    recordComponents="true" numScatteringLevels="0" recordPolarization="false" recordStatistics="true"/>
                    <!--<PerspectiveInstrument instrumentName="perspective" numPixelsX="1920" numPixelsY="1080"
                    width="{fovX}" viewX="40 kpc" viewY="0 pc" viewZ="0 pc" crossX="0 pc" crossY="0 pc" crossZ="0 pc"
                    upX="0 pc" upY="1 pc" upZ="0 pc" focal="40 kpc" recordComponents="false" numScatteringLevels="0"
                    recordPolarization="false" recordStatistics="true"/>-->
                </instruments>
            </InstrumentSystem>
        </instrumentSystem>

        <probeSystem type="ProbeSystem">
            <ProbeSystem>
                <probes type="Probe">
                    <SpatialGridConvergenceProbe probeName="cnv" wavelength="0.55 micron"/>
                    <RadiationFieldPerCellProbe probeName="radfield" writeWavelengthGrid="true"/>
                </probes>
            </ProbeSystem>
        </probeSystem>

    </MonteCarloSimulation>
</skirt-simulation-hierarchy>
"""


def createParticleInputFiles(sP, subhaloID, params, fofScope=False, star_model="fsps", dust_model="const04"):
    """Create the .txt files representing the stellar populations, and gas/dust, to be read in by SKIRT."""
    assert star_model in ["fsps", "bc03mappings3"]
    assert dust_model in ["const04", "var_dtm_z"]
    assert params["simMode"] in ["ExtinctionOnly", "DustEmission", "DustEmissionWithSelfAbsorption"]

    fields_gas = ["pos_rel", "vel_rel", "Density", "GFM_Metallicity", "temp_sfcold_log"]
    fields_stars = ["pos_rel", "vel_rel", "GFM_InitialMass", "GFM_Metallicity", "star_age", "StellarHsml"]

    filename_ski = "sh%d.ski" % subhaloID

    if isfile(filename_ski):
        print("Note: [%s] already exists, skipping input file creation." % filename_ski)
        return filename_ski

    filename_gas = "sh%d_gas_%s_%d_fof=%s_%s.txt" % (subhaloID, sP.simName, sP.snap, fofScope, dust_model)

    # load: rotation
    I = momentOfInertiaTensor(sP, subhaloID=subhaloID)
    rotMatrices = rotationMatricesFromInertiaTensor(I)

    # load
    subhalo = sP.groupCatSingle(subhaloID=subhaloID)
    haloID = None

    if fofScope:
        # switch load to fof-scope
        sP.refPos = subhalo["SubhaloPos"]  # pos_rel load relative to this position
        sP.refVel = subhalo["SubhaloVel"]
        haloID = subhalo["SubhaloGrNr"]
        subhaloID = None

    gas = sP.snapshotSubset("gas", fields_gas, subhaloID=subhaloID, haloID=haloID)
    stars = sP.snapshotSubset("stars_real", fields_stars, subhaloID=subhaloID, haloID=haloID)

    # rotation: we will orient the galaxy, then simultaneously save edge-on, face-on, and random outputs
    pos_stars_rot, _ = rotateCoordinateArray(sP, stars["pos_rel"], rotMatrices["face-on"], [0, 0, 0], shiftBack=False)
    pos_gas_rot, _ = rotateCoordinateArray(sP, gas["pos_rel"], rotMatrices["face-on"], [0, 0, 0], shiftBack=False)

    stars["vel_rel"] = np.asarray(np.transpose(np.dot(rotMatrices["face-on"], stars["vel_rel"].transpose())))
    gas["vel_rel"] = np.asarray(np.transpose(np.dot(rotMatrices["face-on"], gas["vel_rel"].transpose())))

    # coordinates: ckpc/h -> pkpc
    gas["pos_rel"] = sP.units.codeLengthToKpc(pos_gas_rot)
    stars["pos_rel"] = sP.units.codeLengthToKpc(pos_stars_rot)

    # calculate random rotation angle (uniform on the sphere)
    np.random.seed(sP.res + subhaloID + 42)
    azimuth_random = 2 * np.pi * np.random.uniform()
    inclination_random = np.arccos(1 - 2 * np.random.uniform())

    # gas: physical dust density in [msun/pc^3]
    if dust_model == "const04":
        # constant DTM (dust to metal mass ratio) = 0.4
        dtm_ratio = 0.4
        dust_dens = gas["Density"] * gas["GFM_Metallicity"] * dtm_ratio
        dust_dens = sP.units.codeDensToPhys(dust_dens, msunpc3=True)

    if dust_model == "var_dtm_z":
        # variable DTM (dust to metal mass ratio) depending on metallicity, based on Popping+2020

        # calculate the dust fracion of the all metals (Mdust / (Mdust + Mmetal)
        # Based on Remy-Ruyer 2014, variable XCO
        Z_solar = gas["GFM_Metallicity"] / 0.014  # gas phase metallicity in solar units (note: Z_solar value)
        logOH = np.log10(Z_solar) + 8.69
        logGTD = 2.21 + (8.69 - logOH)

        w = np.where(logOH < 8.1)
        logGTD[w] = 0.96 + 3.1 * (8.69 - logOH[w])

        DTG = 1.0 / 10**logGTD  # gas to dust ratio (linear) = M_dust / M_gas
        DTM = DTG / gas["GFM_Metallicity"]  # dust to metal mass ratio (linear) (varies from ~0.03 to ~0.45)

        dust_dens = gas["Density"] * gas["GFM_Metallicity"] * DTM
        dust_dens = sP.units.codeDensToPhys(dust_dens, msunpc3=True)

    if 1:
        # dust sputtering assumption: none if T > 10^4 K, note all SFR>0 gas passes this check
        # note: Vogelsberger+18 uses 8000K, Schulz+20 uses 75000K, Rodriguez-Gomez+19 requires SFR>0
        w = np.where(gas["temp_sfcold_log"] > 4.0)
        dust_dens[w] = 0.0

    # write gas
    data = np.vstack(
        (
            gas["pos_rel"][:, 0],
            gas["pos_rel"][:, 1],
            gas["pos_rel"][:, 2],
            dust_dens,
            gas["vel_rel"][:, 0],
            gas["vel_rel"][:, 1],
            gas["vel_rel"][:, 2],
        )
    ).T

    header = "\ncolumn 1: position x (kpc)\ncolumn 2: position y (kpc)\ncolumn 3: position z (kpc)\n"
    header += "column 4: mass density (Msun/pc3)\n"
    header += "column 5: velocity x (km/s)\ncolumn 6: velocity y (km/s)\ncolumn 7: velocity z (km/s)\n"

    np.savetxt(filename_gas, data, fmt="%g", header=header, newline="\n")
    print("Wrote: [%s] with [%d] gas cells." % (filename_gas, dust_dens.size))

    # stars
    if star_model == "bc03mappings3":
        # separate young and old (10 Myr)
        inds_young = np.where(stars["star_age"] <= 0.01)[0]
        inds_old = np.where(stars["star_age"] > 0.01)[0]

        # mappings-III parameters
        t_clear = 1e7  # yr
        logC = 5.0
        log_P_over_k = 5.0  # log10(P/KB [K/cm^3])
        P0 = 1e6 * 10.0**log_P_over_k * sP.units.boltzmann_JK  # J/cm^3 -> J/m^3 = linear Pa
        fPDR = 0.2

        # write young
        data = []

        if len(inds_young):
            data = np.vstack(
                (
                    stars["pos_rel"][inds_young, 0],
                    stars["pos_rel"][inds_young, 1],
                    stars["pos_rel"][inds_young, 2],
                    sP.units.codeLengthToKpc(stars["StellarHsml"][inds_young]),
                    stars["vel_rel"][inds_young, 0],
                    stars["vel_rel"][inds_young, 1],
                    stars["vel_rel"][inds_young, 2],
                    sP.units.codeMassToMsun(stars["GFM_InitialMass"][inds_young]) / t_clear,
                    stars["GFM_Metallicity"][inds_young],
                    np.zeros(inds_young.size) + logC,
                    np.zeros(inds_young.size) + P0,
                    np.zeros(inds_young.size) + fPDR,
                )
            ).T

        filename_stars1 = "sh%d_stars_young_%s_%d_fof=%s_%s.txt" % (
            subhaloID,
            sP.simName,
            sP.snap,
            fofScope,
            star_model,
        )
        header = "\ncolumn 1: position x (kpc)\ncolumn 2: position y (kpc)\ncolumn 3: position z (kpc)\n"
        header += "column 4: size h (kpc)\ncolumn 5: velocity x (km/s)\ncolumn 6: velocity y (km/s)\n"
        header += "column 7: velocity z (km/s)\ncolumn 8: SFR (Msun/yr)\n"
        header += "column 9: metallicity (1)\ncolumn 10: logC (1)\ncolumn 11: P0 (Pa)\ncolumn 12: fPDR (1)\n"

        np.savetxt(filename_stars1, data, fmt="%.10g", header=header, newline="\n")
        print("Wrote: [%s] with [%d] stars." % (filename_stars1, inds_young.size))

        # write old
        data = []

        if len(inds_old):
            data = np.vstack(
                (
                    stars["pos_rel"][inds_old, 0],
                    stars["pos_rel"][inds_old, 1],
                    stars["pos_rel"][inds_old, 2],
                    sP.units.codeLengthToKpc(stars["StellarHsml"][inds_old]),
                    stars["vel_rel"][inds_old, 0],
                    stars["vel_rel"][inds_old, 1],
                    stars["vel_rel"][inds_old, 2],
                    sP.units.codeMassToMsun(stars["GFM_InitialMass"][inds_old]),
                    stars["GFM_Metallicity"][inds_old],
                    stars["star_age"][inds_old],
                )
            ).T

        filename_stars2 = "sh%d_stars_old_%s_%d_fof=%s_%s.txt" % (subhaloID, sP.simName, sP.snap, fofScope, star_model)
        header = "\ncolumn 1: position x (kpc)\ncolumn 2: position y (kpc)\ncolumn 3: position z (kpc)\n"
        header += "column 4: size h (kpc)\n"
        header += "column 5: velocity x (km/s)\ncolumn 6: velocity y (km/s)\ncolumn 7: velocity z (km/s)\n"
        header += "column 8: initial mass (Msun)\ncolumn 9: metallicity (1)\ncolumn 10: age (Gyr)\n"

        np.savetxt(filename_stars2, data, fmt="%.10g", header=header, newline="\n")
        print("Wrote: [%s] with [%d] stars." % (filename_stars2, inds_old.size))

        # fill SKI template (star_model specific)
        starsSourceModel = stars_source_bc03mappings3.format(
            filenameStars1=filename_stars1, filenameStars2=filename_stars2
        )

    if star_model == "fsps":
        # write stars
        data = np.vstack(
            (
                stars["pos_rel"][:, 0],
                stars["pos_rel"][:, 1],
                stars["pos_rel"][:, 2],
                sP.units.codeLengthToKpc(stars["StellarHsml"]),
                stars["vel_rel"][:, 0],
                stars["vel_rel"][:, 1],
                stars["vel_rel"][:, 2],
                sP.units.codeMassToMsun(stars["GFM_InitialMass"]),
                stars["GFM_Metallicity"],
                stars["star_age"],
            )
        ).T

        filename_stars = "sh%d_stars_%s_%d_fof=%s_%s.txt" % (subhaloID, sP.simName, sP.snap, fofScope, star_model)
        header = "\ncolumn 1: position x (kpc)\ncolumn 2: position y (kpc)\ncolumn 3: position z (kpc)\n"
        header += "column 4: size h (kpc)\n"
        header += "column 5: velocity x (km/s)\ncolumn 6: velocity y (km/s)\ncolumn 7: velocity z (km/s)\n"
        header += "column 8: initial mass (Msun)\ncolumn 9: metallicity (1)\ncolumn 10: age (Gyr)\n"

        np.savetxt(filename_stars, data, fmt="%.10g", header=header, newline="\n")
        print("Wrote: [%s] with [%d] stars." % (filename_stars, stars["star_age"].size))

        # fill SKI template (star_model specific)
        starsSourceModel = stars_source_fsps.format(filenameStars=filename_stars)

    # fill SKI template
    if params["simMode"] == "ExtinctionOnly":
        dustModelParameters = dust_extinction.format()

    if params["simMode"] in ["DustEmission", "DustEmissionWithSelfAbsorption"]:
        dustModelParameters = dust_emission.format(
            numWavelengths=params["numWavelengths"],
            minWavelength=params["minWavelength"],
            maxWavelength=params["maxWavelength"],
        )
    if params["simMode"] == "DustEmissionWithSelfAbsorption":
        dustModelParameters += dust_selfabs

    ski_contents = ski_template.format(
        starsSourceModel=starsSourceModel,
        dustModelParameters=dustModelParameters,
        minX="%d kpc" % np.floor(gas["pos_rel"][:, 0].min()),
        maxX="%d kpc" % np.ceil(gas["pos_rel"][:, 0].max()),
        minY="%d kpc" % np.floor(gas["pos_rel"][:, 1].min()),
        maxY="%d kpc" % np.ceil(gas["pos_rel"][:, 1].max()),
        minZ="%d kpc" % np.floor(gas["pos_rel"][:, 2].min()),
        maxZ="%d kpc" % np.ceil(gas["pos_rel"][:, 2].max()),
        filenameGas=filename_gas,
        simMode=params["simMode"],
        numPackets=params["numPackets"],
        numPixelsX=params["numPixelsX"],
        numPixelsY=params["numPixelsY"],
        fovX=params["fovX"],
        fovY=params["fovY"],
        numWavelengths=params["numWavelengths"],
        minWavelength=params["minWavelength"],
        maxWavelength=params["maxWavelength"],
        incRandom="%d deg" % np.rad2deg(inclination_random),
        aziRandom="%d deg" % np.rad2deg(azimuth_random),
    )

    # write SKI configuration file
    with open(filename_ski, "w") as f:
        f.write(ski_contents)

    print("Wrote: [%s]" % filename_ski)

    return filename_ski


def driver_single():
    """Prepare and execute a single SKIRT run of one subhalo.

    Note that all input and output files are created in the working directory, and left there after for visualization.
    By default, three projections are made simultaneously: edge-on, face-on, and one random orientation.
    """
    sP = simParams(run="tng100-1", redshift=0.05)
    subhaloID = 343503  # almost edge-on disk

    dust_model = "var_dtm_z"  #'const04' # const04, var_dtm_z
    star_model = "fsps"  #'bc03mappings3' # fsps, bc03mappings3
    fofScope = False

    # use ExtinctionOnly for UV+Opt+NIR. If we want IR dust emission, should switch to
    # DustEmissionWithSelfAbsorption and increase maxWavelength to ~100 microns.
    # note: radiation field per cell probe only output for DustEmission* sim modes.
    params = {
        "numPackets": 1e6,
        "numPixelsX": 200,
        "numPixelsY": 200,
        "fovX": "40 kpc",
        "fovY": "40 kpc",
        "numWavelengths": 40,
        "simMode": "ExtinctionOnly",  # ExtinctionOnly, DustEmission, DustEmissionWithSelfAbsorption
        "minWavelength": "0.15 micron",  # 1500 Ang
        "maxWavelength": "1.0 micron",
    }  # 10000 Ang

    # create inputs
    filename_ski = createParticleInputFiles(
        sP, subhaloID, params, fofScope=fofScope, dust_model=dust_model, star_model=star_model
    )

    # run already finished?
    if isfile("sh%d_random_total.fits" % subhaloID):
        print("Note: Output already done, skipping.")
        return

    # execute SKIRT (multi-threaded, use all cores on this machine by default)
    print("Running SKIRT...")
    start_time = time.time()

    p = subprocess.run(["skirt", filename_ski], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    if p.returncode != 0:
        print(p.stdout.decode("utf8"))
        print("\nSKIRT RUN FAILED!")
    else:
        lines = p.stdout.decode("utf8").split("\n")
        for line in lines[-5:]:
            if "threads" in line or "memory" in line:
                print(" ", line)

    print("Done, took [%.1f] sec." % (time.time() - start_time))


def driver_sample(sP, subhaloIDs):
    """Prepare and execute a large number of jobs."""
    dust_model = "const04"  # const04, var_dtm_z
    star_model = "fsps"  #'bc03mappings3' # fsps, bc03mappings3
    fofScope = False

    maxSimultaneousJobs = 4

    # use ExtinctionOnly for UV+Opt+NIR. If we want IR dust emission, should switch to
    # DustEmissionWithSelfAbsorption and increase maxWavelength to ~100 microns.
    params = {
        "numPackets": 5e6,
        "numPixelsX": 400,
        "numPixelsY": 400,
        "fovX": "40 kpc",
        "fovY": "40 kpc",
        "numWavelengths": 100,
        "simMode": "ExtinctionOnly",  # ExtinctionOnly, DustEmission, DustEmissionWithSelfAbsorption
        "minWavelength": "0.09 micron",
        "maxWavelength": "1.0 micron",
    }

    # create inputs
    for subhaloID in subhaloIDs:
        filename_ski = createParticleInputFiles(
            sP, subhaloID, params, fofScope=fofScope, dust_model=dust_model, star_model=star_model
        )
        assert filename_ski == "sh%d.ski" % subhaloID  # assumed in job script

    # create job [array] file for submission, one job per subhalo, one full node (threaded) per job
    jobFile = """#!/bin/sh
#SBATCH --mail-user dnelson@uni-heidelberg.de
#SBATCH --mail-type=FAIL #,ARRAY_TASKS
#SBATCH -p p.24h
#SBATCH --array=0-{numJobs}%{maxSimultaneousJobs}
#SBATCH -J skirt
#SBATCH -o run_%a.txt
#SBATCH -e run_%a.err
#SBATCH -t 01:00:00 # one hour
#SBATCH --mem=190000 # normal memory nodes
#SBATCH --exclusive
#SBATCH --ntasks 40
#SBATCH --ntasks-per-node=40

subhaloIDs=({subIDs})
skifilename="sh${{subhaloIDs[${{SLURM_ARRAY_TASK_ID}}]}}.ski"

skirt -t $SLURM_NTASKS_PER_NODE $skifilename
    """

    jobFile = jobFile.format(
        numJobs=(len(subhaloIDs) - 1),
        maxSimultaneousJobs=maxSimultaneousJobs,
        subIDs=" ".join([str(subID) for subID in subhaloIDs]),
    )

    with open("job.slurm", "w") as f:
        f.write(jobFile)

    print('Wrote: [job.slurm], suggest to now execute with "sbatch job.slurm".')


def _log(x):
    """Return natural log for positive values, large negative number for zero negative values."""
    r = np.zeros(x.shape, dtype=x.dtype)
    w = x > 0
    r[w] = np.log(x[w])
    r[~w] = -750.0
    return r


def convolve_cube_with_filter(data, wavelengths, filterName, raw=False):
    """Return convolution of IFU (x,y,wave) cube with a given broadband.

    Args:
        data (ndarray[float]): 3D array (x,y,wavelength) of flux values.
        wavelengths (ndarray[float]): wavelengths corresponding to the 3rd axis of data [micron].
        filterName (str): name of the filter, e.g. 'sdss_g'.
        raw (bool): if True, do not do any unit conversions (e.g. for stats0 analysis).
    """
    # get broadband filter and normalize
    import fsps

    band = fsps.get_filter(filterName)
    filter_wave_micron = band.transmission[0] * 1e-4  # ang -> micron
    filter_trans_norm = band.transmission[1]
    filter_trans_norm /= np.trapz(x=filter_wave_micron, y=band.transmission[1])  # 1/micron

    # convert 'natural' flux unit into 'per wavelength'
    if not raw:
        data /= wavelengths

    # common wavelength grid: assume that SKIRT run covers the filter
    w = np.where((wavelengths >= filter_wave_micron[0]) & (wavelengths < filter_wave_micron[-1]))

    if len(w[0]) == 0:
        return None

    wave_loc = wavelengths[w]
    data_loc = data[:, :, w[0]]

    # log-log interpolate transmission onto this common grid
    log_trans = _log(filter_trans_norm)
    T = np.exp(np.interp(np.log10(wave_loc), np.log10(filter_wave_micron), log_trans, left=0.0, right=0.0))

    F = data_loc
    result = np.trapz(x=wave_loc, y=F * T)

    # convert 'per wavelength' back into 'natural' flux units
    if not raw:
        pivot_wavelength = np.sqrt(1.0 / np.trapz(x=filter_wave_micron, y=filter_trans_norm / filter_wave_micron**2))
        result *= pivot_wavelength

    return result


def vis(subhaloID=343503):
    """Load and visualize SKIRT output.

    This routine simply looks for files in the current  working directory corresponding to the input subhalo ID.
    """
    instNames = ["faceon", "edgeon", "random", "perspective"]
    outTypes = ["total", "transparent"]

    filterNames = ["sdss_g", "sdss_r", "sdss_i"]

    redshift = 0.0  # if >0, add surface-brightness dimming to this redshift

    # loop over instruments/output types
    for instName in instNames:
        for outType in outTypes:
            filename = "sh%d_%s_%s.fits" % (subhaloID, instName, outType)

            if not isfile(filename):
                continue
            print(filename)

            # load
            with fits.open(filename) as f:
                header = f[0].header
                data = np.transpose(f[0].data, [2, 1, 0])

                wavelengths = f["Z-axis coordinate values"].data["GRID_POINTS"]

            # no automatic unit conversion, so make sure we have the expected values
            assert data.shape[0] == header["NAXIS1"] and data.shape[1] == header["NAXIS2"]  # x,y,wavelength
            assert header["CUNIT1"] == "pc" and header["CUNIT2"] == "pc" and header["CUNIT3"] == "micron"
            assert header["BUNIT"] == "W/m2/arcsec2"

            # create spatial axes
            start_x = header["CRVAL1"] - header["CDELT1"] * (header["CRPIX1"] - 1)
            start_y = header["CRVAL2"] - header["CDELT2"] * (header["CRPIX2"] - 1)
            stop_x = header["CRVAL1"] + header["CDELT1"] * (header["NAXIS1"] - header["CRPIX1"])
            stop_y = header["CRVAL2"] + header["CDELT2"] * (header["NAXIS2"] - header["CRPIX2"])

            xvals = np.linspace(start_x, stop_x, header["NAXIS1"]) / 1e3  # kpc
            yvals = np.linspace(start_y, stop_y, header["NAXIS2"]) / 1e3  # kpc

            images = []

            # loop over all filters
            for filterName in filterNames:
                # create a band image (original BUNIT)
                convolved = convolve_cube_with_filter(data, wavelengths, filterName)

                # surface brightness dimming
                if redshift > 0:
                    # exponent is 4 for integrated/bolometric flux
                    # 5 for flux per wavelength interval
                    # 3 for flux per frequency interval
                    convolved /= (1 + redshift) ** 4

                # units: [W/m^2/arcsec^2] -> [erg/s/cm^2/arcsec^2]
                convolved *= 1e7 / 1e4  # J/s -> erg/s, m^-2 -> cm^-2

                # create image and save
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111)

                im = logZeroMin(np.flip(convolved.copy(), axis=1))
                plt.imshow(im, cmap="magma", aspect="equal", extent=[xvals[0], xvals[-1], yvals[0], yvals[-1]])
                ax.autoscale(False)
                plt.clim([-15.0, -12.0])

                ax.set_xlabel("x [kpc]")
                ax.set_ylabel("y [kpc]")

                cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.15)
                cb = plt.colorbar(cax=cax)
                cb.ax.set_ylabel(r"Surface Brightness [ log erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$ ]")

                fig.savefig("image_sh%d_%s_%s_%s.pdf" % (subhaloID, outType, instName, filterName))
                plt.close(fig)

                # save
                images.append(convolved)

            # loaded three bands? then create composite RGB now
            if len(images) == 3:
                rgb = np.zeros((images[0].shape[0], images[0].shape[1], 3), dtype="float32")

                def logstretch(x, a):
                    return np.log(a * x + 1) / np.log(a + 1)

                # stamp in individual band images (for collective bounds)
                for i in range(3):
                    rgb[:, :, -(i + 1)] = images[i]

                # can derive limits either on rgb as a whole, or on each band
                w = np.where(rgb > 0.0)
                minval = np.percentile(rgb[w], 20)  # images[0].min()
                maxval = np.percentile(rgb[w], 99.5)  # images[0].max()

                # normalize
                for i in range(3):
                    grid_norm = (images[i] - minval) / (maxval - minval)

                    rgb[:, :, -(i + 1)] = grid_norm

                # clip and scale
                rgb = np.clip(rgb, 0.0, 1.0)
                rgb = logstretch(rgb, a=1000)

                # save
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111)

                plt.imshow(rgb, aspect="equal", origin="lower", extent=[xvals[0], xvals[-1], yvals[0], yvals[-1]])

                ax.set_xlabel("x [kpc]")
                ax.set_ylabel("y [kpc]")

                fig.savefig("image_sh%d_%s_%s_rgb.pdf" % (subhaloID, outType, instName))
                plt.close(fig)

        # stats?
        statsfile = filename = "sh%d_%s_stats1.fits" % (subhaloID, instName)
        if isfile(statsfile):
            print(statsfile)
            # w_i is the contribution of the i^th photon packet to each pixel
            # includes contributions of all photon packets peeled-off and/or scattered from
            # originally launched packets
            with fits.open(statsfile) as f:
                header = f[0].header  # sum of w_i^k for k=1
                wi1 = np.transpose(f[0].data, [2, 1, 0])

            with fits.open(statsfile.replace("stats1", "stats2")) as f:
                header = f[0].header  # sum of w_i^k for k=2
                wi2 = np.transpose(f[0].data, [2, 1, 0])

            # need photon packet count
            with open("sh%d.ski" % subhaloID) as f:
                lines = f.readlines()

            for line in lines:
                if "numPackets" in line:
                    N = float(line.split('numPackets="')[1].split('">')[0])

            # derive relative err (see Camps & Baes 2018 Eqn 14)
            with np.errstate(invalid="ignore"):
                rel_err = np.sqrt(wi2 / wi1**2 - 1 / N)

            rel_err_band = convolve_cube_with_filter(rel_err, wavelengths, filterNames[0], raw=True)

            # create image and save
            fig, ax = plt.subplots()

            im = np.flip(rel_err_band, axis=1)
            plt.imshow(im, cmap="tab10", aspect="equal", extent=[xvals[0], xvals[-1], yvals[0], yvals[-1]])
            ax.autoscale(False)
            plt.clim([0.0, 1.0])

            ax.set_xlabel("x [kpc]")
            ax.set_ylabel("y [kpc]")

            cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.15)
            cb = plt.colorbar(cax=cax)
            cb.ax.set_ylabel(r"Relative Error R = $[ \Sigma w_i^2 / (\Sigma w_i)^2 - 1 / N ]^{1/2}$")

            fig.savefig("image_sh%d_%s_%s_relerr.pdf" % (subhaloID, instName, filterName))
            plt.close(fig)


def plot_sed(subhaloID=343503):
    """Plot a spectral energy distribution .dat output file.

    Looks for existing output files in the current working directory.
    """
    instNames = ["faceon", "edgeon", "random"]

    for instName in instNames:
        filename = "sh%d_%s_sed.dat" % (subhaloID, instName)

        print(filename)

        # column 1: wavelength (micron)
        # column 2: total flux; lambda*F_lambda (W/m2)
        # column 3: transparent flux; lambda*F_lambda (W/m2)
        # column 4: direct primary flux; lambda*F_lambda (W/m2)
        # column 5: scattered primary flux; lambda*F_lambda (W/m2)
        # column 6: direct secondary flux; lambda*F_lambda (W/m2)
        # column 7: scattered secondary flux; lambda*F_lambda (W/m2)
        data = np.loadtxt(filename)

        labels = {
            1: "total",
            2: "transparent",
            3: "direct primary",
            4: "scattered primary",
            5: "direct secondary",
            6: "scattered secondary",
        }

        # start plot (full sed)
        fig, ax = plt.subplots()

        ax.set_xlim(np.log10([0.09, 100]))
        ax.set_ylim([-22, -10])
        ax.set_xlabel(r"$\lambda$ [log micron]")
        ax.set_ylabel(r"$\lambda$ F$_{\lambda}$ [W m$^{-2}$]")

        xx = np.log10(data[:, 0])

        for col, label in labels.items():
            ax.plot(xx, logZeroNaN(data[:, col]), "-", label=label)

        ax.legend(loc="best")
        fig.savefig("sed_sh%d_%s.pdf" % (subhaloID, instName))
        plt.close(fig)

        # start plot (optical)
        fig, ax = plt.subplots()

        ww = np.where(data[:, 0] < 1.0)

        ax.set_xlim([900, 10000])
        ax.set_ylim([-13, -10])
        ax.set_xlabel(r"$\lambda$ [Angstrom]")
        ax.set_ylabel(r"$\lambda$ F$_{\lambda}$ [W m$^{-2}$]")

        xx = data[ww[0], 0] * 1e4  # micron -> ang

        for col, label in labels.items():
            ax.plot(xx, logZeroNaN(data[ww[0], col]), "-", label=label)

        ax.legend(loc="best")
        fig.savefig("sed_optical_sh%d_%s.pdf" % (subhaloID, instName))
        plt.close(fig)
