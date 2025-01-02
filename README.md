# A tutorial on Scalable System Simulations
*Looking at RISC-V architectures and performance analysis for machine learning workloads*

Dive into the world of system-level simulations!
- Explore RISC-V modelling and workload representation using gem5+MLIR.
- Learn to scale your system simulations effortlessly with the power of SST.

This tutorial bridges cutting-edge open-source tools and techniques to empower your hardware-software co-design journey.

If you use this repository, please cite it as follows:

```bibtex
@misc{sim-learning-tutorial,
  author       = {Erwan Lenormand, Tommaso Marinelli, Debjyoti Bhattacharjee},
  title        = {A tutorial on Scalable System Simulations},
  year         = {2025},
  version      = {v1.0.0},
  howpublished = {Presented at the International Conference on VLSI Design 2025},
  note         = {\url{https://github.com/CSA-infra/RISCV-Scaleable-Simulation-tutorial/}  Accessed: 2025-01-02}
}
```

### Getting Started
Build the documentation to get start with the two part tutorial. Minimum required version of python is 3.10
```
python3 -m venv imec_tut_2025
source imec_tut_2025/bin/activate
pip3 install -r requirements.txt
make html
```

### Need help?
If you need help or clarification regarding any part of the tutorial, file an issue in the repository.