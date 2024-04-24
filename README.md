<!-- todo! DOI badge -->

# Unified Momentum Model
This repository contains a reference implementation of the Unified Momentum Model presented in the associated manuscript titled **Unified Momentum Model for Rotor Aerodynamics Across Operating Regimes**. The Unified Momentum Model is a novel approach for modeling a yaw-misaligned actuator disk induction and outlet velocities, including negative and high thrust regimes. 

It builds on the existing actuator disk model described in [Heck, K. S., Johlas, H. M., & Howland, M. F. (2023). ***Modelling the induction, thrust and power of a yaw-misaligned actuator disk***. Journal of Fluid Mechanics](https://doi.org/10.1017/jfm.2023.129).




# Installation
To install this Python package follow one of the following methods.
### Direct installation from Github
To install directly from Github into the current Python environment, run:
```bash
pip install git+https://github.com/Howland-Lab/Unified-Momentum-Model.git
```


### Install from cloned repository
If you prefer to download the repository first (for example, to run the example and paper figure scripts), you can first clone the repository, either using http:
```bash
git clone https://github.com/Howland-Lab/Unified-Momentum-Model.git
```
or ssh:
```bash
git clone git@github.com:Howland-Lab/Unified-Momentum-Model.git
```
then, install locally using pip using `pip install .` for the base installation, or `pip install .[figures]` to install the extra dependencies required to run the examples and paper figure scripts and notebooks:

```
cd Unified-Momentum-Model
pip install .[figures]
```
# Usage
This repository contains 
1) a Python package which implements the Unified Momentum Model (see the `UnifiedMomentumModel` folder)
2) working example scripts using the package in Python (see the `examples` folder)
3) Jupyter notebooks which recreate all figures in the manuscript (see the `figures` folder).
## Package usage
A `UnifiedMomentum` model object can be instantiated and can be called to solve the actuator disk model model for a given local thrust coefficient, $C_T'$ and rotor yaw angle in radians, $\gamma$. Here is a short python script which demonstrates this:

```python
from UnifiedMomentumModel.Momentum import UnifiedMomentum

model = UnifiedMomentum()
solution = model(Ctprime=2.0, yaw=0.0)

print(f"rotor-normal induction factor: {solution.an}")
print(f"streamwise outlet velocity: {solution.u4}")
print(f"lateral outlet velocity: {solution.v4}")
print(f"near-wake length: {solution.x0}")
print(f"outlet pressure: {solution.dp}")
print(f"Rotor power coefficient: {solution.Cp}")
print(f"Rotor thrust coefficient: {solution.Ct}")
```



# Contributions
If you have suggestions or issues with the Unified Momentum Model, feel free to raise an issue or submit a pull request.

<!-- # Citation
If you want to cite the Unified Momentum Model, please use this citation:
To do -->