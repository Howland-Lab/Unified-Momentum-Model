<!-- todo! DOI badge -->

# Unified Momentum Model
This repository contains a reference implementation of the Unified Momentum Model presented in [**paper title**] published in [**journal name**]. The Unified Momentum Model is a novel approach for modeling a yaw-misaligned actuator disk induction and outlet velocities, including negative and high thrust regimes. 

It builds on the existing actuator disk model described in [Heck, K. S., Johlas, H. M., & Howland, M. F. (2023). ***Modelling the induction, thrust and power of a yaw-misaligned actuator disk***. Journal of Fluid Mechanics](https://doi.org/10.1017/jfm.2023.129).




# Installation
To install this Python package, clone this repository and pip install:
```bash
git clone git@github.com:Howland-Lab/Unified-Momentum-Model.git
cd Unified-Momentum-Model
pip install .
```
# Usage
A `UnifiedMomentum` model object can be instantiated and can be called to solve the model for a given thrust coefficient and rotor yaw angle. Here is a short python script which demonstrates this:

```python
from UnifiedMomentumModel.Momentum import UnifiedMomentum

model = UnifiedMomentum()
solution = model.solve(Ctprime=2.0, yaw=0.0)

print(f"induction: {solution.an}")
print(f"streamwise outlet velocity: {solution.u4}")
print(f"lateral outlet velocity: {solution.v4}")
print(f"near-wake length: {solution.x0}")
print(f"outlet pressure: {solution.dp}")
```



# Contributions
If you have suggestions or issues with the Unified Momentum Model, feel free to raise an issue or submit a pull request.

# Citation
If you want to cite the Unified Momentum Model, please use this citation:
To do