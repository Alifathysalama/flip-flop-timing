# Flip-flop timing simulations

This repository contains an implementation of a D flip-flop with async
active-low reset using the
[IHP-Open-PDK](https://github.com/IHP-GmbH/IHP-Open-PDK) 130nm technology. Some
ngspice simulations are done to measure the timing properties of the flip-flop.

## Setup

1. Ensure that git submodules are checked out, either by having run `git clone
   --recursive` or by running:
   
```
git submodules init
git submodules update
```

2. Compile the OpenVAF models for ngspice simulation. This requires
   [OpenVAF-Reloaded](https://github.com/OpenVAF/OpenVAF-Reloaded), and the
   `openvaf` command to exist (OpenVAF-Reloaded is called `openvaf-r`, so an
   alias or symlink must exist).

```
cd ./IHP-Open-PDK/ihp-sg13g2/libs.tech/verilog-a/
./openvaf-compile-va.sh
cd -
```

## Usage

[Just](https://github.com/casey/just) can be used to run commonly used tools:

- `just magic`. Run [Magic](http://opencircuitdesign.com/magic/) and open the
  flip-flop cell. This downloads a Magic AppImage if it has not been downloaded
  previously.

- `just extract`. Run Magic to extract the flip-flop circuit to a ngspice
  netlist.

- `just ngspice`. Run a basic ngspice simulation that can be used as a unit test
  that checks that the flip-flop is implemented correctly. This runs `just
  extract` to update the extracted ngspice netlist before running the
  simulation. The simulation output can be plotted with the
  [ngspice unit test plots](./notebooks/ngpsice unit test plots.ipynb) Jupyter
  notebook.

Additionally, there is an
[ngpsice timing simulations](./notebooks/ngspice timing simulations.ipynb)
notebook that is used to run and plot ngspice simulations for timing analysis
parameters.

Before running any other tools manually, generally the following needs to be run
to set up env variables for the PDK:

```
source setup-pdk
```

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
