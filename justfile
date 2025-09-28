set export

magic_appimage := "https://github.com/RTimothyEdwards/magic/releases/download/8.3.551/Magic-8.3.551.20250911.00e3bbd-x86_64-EL10.AppImage"

cell := "fdc_dense"

download-magic:
    #!/usr/bin/env bash
    if [ ! -f $(basename $magic_appimage) ]; then
        wget $magic_appimage
        chmod +x $(basename $magic_appimage)
    fi

magic: download-magic
    #!/usr/bin/env bash
    source setup-pdk
    cd magic
    ../$(basename $magic_appimage) -d XR \
        -rcfile ${PDK_ROOT}/ihp-sg13g2/libs.tech/magic/ihp-sg13g2.magicrc ${cell}.mag

extract: download-magic
    #!/usr/bin/env bash
    source setup-pdk
    cd magic
    ../$(basename $magic_appimage) \
        -rcfile ${PDK_ROOT}/ihp-sg13g2/libs.tech/magic/ihp-sg13g2.magicrc -dnull magic-extract.tcl

ngspice: extract
    #!/usr/bin/env bash
    source setup-pdk
    ngspice ngspice/sim.spice
