# Source: cern_electron_collision
**Category:** science_engineering
**README generated:** 2025-10-02 12:52:18Z

## Source URL(s)
- https://www.kaggle.com/datasets/fedesoriano/cern-electron-collision-data

## Citation(s)
- McCauley, Thomas; (2014). Events with two electrons from 2010. CERN Open Data Portal. DOI:10.7483/OPENDATA.CMS.PCSW.AHVG

## Variables (X → Y)
[Filtered by: Q2, Q1]

- energies(E1,E2) → M
- transverse momenta(pt1,pt2) → M

## Causal reasoning

The invariant mass (M) is a derived quantity of the two-electron system, calculated from their four-momenta as M = sqrt((E1 + E2)^2 - (px1 + px2)^2 - (py1 + py2)^2 - (pz1 + pz2)^2), or equivalently expressed using transverse momentum, pseudorapidity, and azimuthal angle as M = sqrt(2 * pt1 * pt2 * (cosh(eta1 - eta2) - cos(phi1 - phi2))); it is causally determined by the electrons’ energies and momentum components (E, px, py, pz) or, equivalently, by their transverse momenta (pt1, pt2), pseudorapidities (eta1, eta2), and azimuthal angles (phi1, phi2), which together define their four-momenta.








