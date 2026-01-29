"""
Metadata for (original) snapshot fields.
"""

from .snapshot import snapshot_fields


# --- general/all particle types ---

masses = {"label": "Mass", "units": "code_mass", "limits": [-3.0, 0.0], "limits_halo": [-6.0, -2.0], "log": True}

# --- gas ---

density = {
    "label": r"$\rho_{\rm gas}$",
    "units": "code_density",
    "limits": [-12.0, 0.0],
    "limits_halo": [-4.0, 2.0],
    "log": True,
}

starformationrate = {
    "label": "Star Formation Rate",
    "units": r"M$_{\rm sun}$ yr$^{-1}$",
    "limits": [-4.0, 2.0],
    "log": True,
}

# add all entries into the snapshot_fields dict
fields = list(locals())

for field in fields:
    # skip non-fields
    if field.startswith("__") or field in ["np", "snapshot_fields"]:
        continue

    # add to registry
    snapshot_fields[field] = locals()[field]
