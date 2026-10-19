"""Published simulation data from papers, downloaded on demand for the reproduction scripts in
``examples/papers/``.

>>> from ewald import examples, import_file
>>> folder = examples.fetch("li6ps5cl-aimd")          # downloads once, then uses the cache
>>> pipe = import_file(folder / "XDATCAR.gz")

Every file is pinned to a commit of the authors' GitHub repository and checked against its
SHA-256, so a script always sees the bytes it was written for. The files are not shipped with
ewald (their licenses and sizes differ); they land in ``<cache dir>/papers/<name>/``
(``EWALD_CACHE_DIR`` moves the cache). ``ewald examples --papers`` lists them and
``ewald examples --fetch NAME`` downloads one.
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RemoteFile:
    name: str      # local file name
    url: str       # pinned to a commit
    sha256: str
    size: int      # bytes


@dataclass(frozen=True)
class Paper:
    name: str
    title: str
    citation: str
    doi: str
    source: str        # repository the files come from
    license: str
    description: str   # system, what the files hold
    reproduces: str    # figures/numbers and the ewald modifiers that reproduce them
    script: str        # file in examples/papers/
    files: tuple[RemoteFile, ...]

    @property
    def size(self) -> int:
        return sum(f.size for f in self.files)


def _github(repo: str, commit: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"


_KINISI = ("bjmorgan/kinisi", "6f4ca50baad855d0e25f049bc575b419edf44861")
_CARBON = ("jla-gardner/carbon-data", "ba0c3a9e16a4ef1a12adcac98bd97764b7e194ea")
_ASI = ("lamr18/aSi-data", "dcbc6ab107702742a41460a06e18014f3f376bb0")

_LLZO_XTC = {
    500: ("2f8380469e72aa2bd48681da615491fca94ee1d3a6c29c1304e59d0f00408d79", 7147840),
    600: ("2c2d8b4722862d120de179b3fdac977cb3bd5e1e639ef9c1e29f2b5e9a1ca126", 7100800),
    700: ("d520d1284df26c3d79373e4eb7585e65dfefa29e7b0f71fb4de8ae7b602e070a", 7084268),
    800: ("80696846ed7d6a2cc506ab5311c135fd33a0296be24038d2bcf8d19a72f99dcc", 7108456),
}
_LLZO_GRO = ("3e2f2af9ed7b7a69d8a255e77d002d4f3f76dceefaf63c52607c35224fc42361", 69179)

_CARBON_FILES = {   # density in g/cm3: (sha256, size), anneal temperature 3000 K
    "1.0": ("6a1562d1fd3a8790a49e3599f7c8353aad51494b271c01802ef7bcac0e8272d3", 5194288),
    "1.5": ("462f8fbe132efeb91adfd166808a98c11c1ecdc82d066699e5de65c6a913e166", 5194288),
    "2.0": ("bd8d327921d9c408e674ed93053e0cbacb5a10d62cb186e788f90fad559ec309", 5193655),
    "2.5": ("a624d61da1d6d2972c61e58046025496af020c61d6abee545ebd1cf740e7d7d4", 5194288),
    "3.0": ("65f9b5bec2b75c869fc4cf7607c00d7c253aa58c7207caf67513793946b68720", 5194288),
    "3.5": ("adf5dd861b648c21adf6d9b14efa128c7a52f345b0419b2025230524ba2cd2b7", 5194288),
}

PAPERS = [
    Paper(
        name="li6ps5cl-aimd",
        title="Li6PS5Cl argyrodite AIMD (kinisi example)",
        citation="B. J. Morgan, Chem. Mater. 33, 2004 (2021); A. R. McCluskey, S. W. Coles, B. J. Morgan, "
                 "J. Chem. Theory Comput. 21, 79 (2025)",
        doi="10.1021/acs.jctc.4c01249",
        source="https://github.com/bjmorgan/kinisi",
        license="MIT",
        description="VASP XDATCAR, Li192 P32 S160 Cl32 (416 atoms), 140 frames 100 fs apart",
        reproduces="Li mean squared displacement and self-diffusion coefficient D* "
                   "(MeanSquaredDisplacement), compared with kinisi",
        script="li6ps5cl_diffusion.py",
        files=(RemoteFile("XDATCAR.gz", _github(*_KINISI, "docs/source/example_XDATCAR.gz"),
                          "cdb11eb4f490277739fd30fb91af87d0a98956ce1392d56c13822520277973f2", 842426),),
    ),
    Paper(
        name="llzo-arrhenius",
        title="Li7La3Zr2O12 (LLZO) MD at 500-800 K (kinisi example)",
        citation="A. R. McCluskey, S. W. Coles, B. J. Morgan, J. Chem. Theory Comput. 21, 79 (2025)",
        doi="10.1021/acs.jctc.4c01249",
        source="https://github.com/bjmorgan/kinisi",
        license="MIT",
        description="GROMACS XTC + GRO, 1536 atoms, 1000 frames (50.8 ps) at each of 500, 600, 700, 800 K",
        reproduces="Arrhenius plot of the Li diffusion coefficient and its activation energy "
                   "(MeanSquaredDisplacement, IonicConductivity)",
        script="llzo_arrhenius.py",
        files=tuple(
            f for t, (sha, size) in _LLZO_XTC.items() for f in (
                RemoteFile(f"traj_{t}.xtc", _github(*_KINISI, f"docs/source/_static/traj_{t}.xtc"), sha, size),
                RemoteFile(f"traj_{t}.gro", _github(*_KINISI, f"docs/source/_static/traj_{t}.gro"), *_LLZO_GRO))),
    ),
    Paper(
        name="carbon-synth",
        title="Synthetic amorphous carbon, C-GAP-17 melt-quench-anneal (C-SYNTH-23)",
        citation="J. L. A. Gardner, Z. F. Beaulieu, V. L. Deringer, Digital Discovery 2, 651 (2023)",
        doi="10.1039/D2DD00137C",
        source="https://github.com/jla-gardner/carbon-data",
        license="MIT (as listed by load-atoms; the repository has no license file)",
        description="extended XYZ, 200 C atoms, 211 frames 1 ps apart with C-GAP-17 local energies, "
                    "densities 1.0-3.5 g/cm3 annealed at 3000 K (6 of the 546 trajectories)",
        reproduces="Fig. 2a, local energy by coordination (sp/sp2/sp3 with a 1.85 A cutoff), and the "
                   "sp3 fraction against density (CoordinationAnalysis)",
        script="carbon_coordination.py",
        files=tuple(RemoteFile(f"density-{rho}-T-3000.extxyz",
                               _github(*_CARBON, f"results/density-{rho}-T-3000.extxyz"), sha, size)
                    for rho, (sha, size) in _CARBON_FILES.items()),
    ),
    Paper(
        name="asi-paracrystalline",
        title="Paracrystalline and polycrystalline amorphous silicon, 100,000 atoms",
        citation="L. A. M. Rosset, D. A. Drabold, V. L. Deringer, Nat. Commun. 16, 2360 (2025)",
        doi="10.1038/s41467-025-57406-4",
        source="https://github.com/lamr18/aSi-data",
        license="none stated (downloaded from the authors' repository, not redistributed)",
        description="extended XYZ, 100,000 Si atoms each, quenched at 1e11 K/s (paracrystalline) and "
                    "1e10 K/s (polycrystalline); the paper's S(q) curves and Laaziri's experiment as CSV",
        reproduces="Fig. 5, X-ray structure factor against the paper's Debye S(q) and experiment "
                   "(StructureFactor), g(r) and crystalline fraction (CoordinationAnalysis, IdentifyDiamond)",
        script="asi_paracrystalline.py",
        files=(
            RemoteFile("100k-para.xyz", _github(*_ASI, "data/xyz/100k-para.xyz"),
                       "2b2cbc0eea15c660244dd898f871365b70df41f259af8baa1347a60f89b45018", 5400227),
            RemoteFile("100k-poly.xyz", _github(*_ASI, "data/xyz/100k-poly.xyz"),
                       "3f6842a740705ff0a5644f84614f142c1ca9f5cd8da65480b81a6198e1e7be55", 5400227),
            RemoteFile("Sq-para.csv", _github(*_ASI, "data/fig-5/Sq-para.csv"),
                       "93261ba7b275027134dd178f578020a54ea34113f4ccf4f56d2a579f19f51bd4", 5436),
            RemoteFile("Sq-poly.csv", _github(*_ASI, "data/fig-5/Sq-poly.csv"),
                       "c86231a05a7793f9652b5af6403157f21e4f619248d3d352bc35a06437b82503", 5156),
            RemoteFile("Sq_Laaziri.csv", _github(*_ASI, "data/fig-5/Sq_Laaziri.csv"),
                       "bd47e86a2f2561a9cef1d60b4a7c63163fff5489a916abc301ce1d31f9963109", 14024),
        ),
    ),
]


def get(name: str) -> Paper:
    for p in PAPERS:
        if p.name == name:
            return p
    raise KeyError(f"no paper dataset {name!r}; choose one of {[p.name for p in PAPERS]}")


def data_dir() -> Path:
    """Where downloaded paper data is kept (next to the remote-file cache)."""
    from ..io.remote import cache_dir

    return cache_dir().parent / "papers"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path, timeout: float = 60.0) -> None:
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as out:
        while block := resp.read(1 << 20):
            out.write(block)
    os.replace(tmp, dest)


def fetch(name: str, dest=None, quiet: bool = False) -> Path:
    """Download the files of the paper dataset ``name`` (once) and return their folder.

    Files already present with the right checksum are kept; a download whose checksum does not
    match is deleted and raises ``ValueError``.
    """
    paper = get(name)
    folder = Path(dest) if dest is not None else data_dir() / paper.name
    folder.mkdir(parents=True, exist_ok=True)
    for f in paper.files:
        path = folder / f.name
        if path.is_file() and path.stat().st_size == f.size and _sha256(path) == f.sha256:
            continue
        if not quiet:
            print(f"downloading {f.name} ({f.size / 1e6:.1f} MB) from {paper.source}", file=sys.stderr)
        _download(f.url, path)
        if _sha256(path) != f.sha256:
            path.unlink()
            raise ValueError(f"{f.url}: checksum mismatch (the file changed upstream?)")
    return folder
