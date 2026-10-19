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

_ARGY = ("bjmorgan/data_argyrodite_disorder", "70060fcba4422a22c37989ea8b0649fb56bd5687")
_TIOF2 = ("bjmorgan/data_TiOF2", "4bda9656882b876ceaf644caccf133772c720512")
_AB = ("gilgalad33/Ammonia-Borane-Dehydrogenation", "3dfe5cc0ae8995c71210388c8bf90cf24a28fb46")
_ALA = ("spiwokv/metadynminerdata", "977e11c8e7b80137d636a002c001e67166dab449")

_ARGYRODITE = {   # <system>_<disorder>_run<n>: (sha256, size); consecutive segments of one run each
    "Li6PS5Cl_0p_run1": ("b0c8270fb4028ffe010357438d2480269a6b93812bdc2d1723b926ceb3b405d4", 841423),
    "Li6PS5Cl_0p_run2": ("1d69175c70d2d34a103774d1476164e222cde59b1ef6fa80760a4e8ec2cf709f", 841437),
    "Li6PS5Cl_0p_run3": ("c01d417fadbc114ece4b0285b42691ab8061f0873df1e0efa6f71181c7ee88d9", 841588),
    "Li6PS5Cl_0p_run4": ("d84ea5de04beaeb0d3ab35209cf452c59ab07303a4dc282e1340c62b260b106a", 841451),
    "Li6PS5Cl_0p_run5": ("147bc3a0bdaf427819adfeb35b0f4766caf00ed9b0367df1af8d2fe9759054d2", 841384),
    "Li6PS5Cl_50p_run1": ("cdb11eb4f490277739fd30fb91af87d0a98956ce1392d56c13822520277973f2", 842426),
    "Li6PS5Cl_50p_run2": ("3321a3a3322047f156c45fd1bd13d180faac88ff5f6dddac3ca8842502f29ea4", 841900),
    "Li6PS5Cl_50p_run3": ("6cf9c7268ef717dc9bb4cfd3255596c37568c4c6edb15caf1ab16560c32bc6bd", 842022),
    "Li6PS5Cl_50p_run4": ("91856ab6a33047ede13bacb2ceeeb43b78f573fe2e183140a9381bfb4b9bdd5a", 841877),
    "Li6PS5Cl_50p_run5": ("2c6e31826330286ba3cb16d5ba477512c18e9d04f80ee60fe4188f212893c334", 421754),
    "Li6PS5Cl_50p_run6": ("f5939541a13299552e52537605de04adbd8bcfc067d7fbefaea2218b45a094e3", 421782),
    "Li6PS5Cl_100p_run1": ("dcd89f5479eaf28e7570995c3661fbbc5cb3015264994ecc775c6e875a43c99c", 841170),
    "Li6PS5Cl_100p_run2": ("e512ac511189205d2a437ec08a6bbb7fae35e267ed5fc1da481097277723de24", 841246),
    "Li6PS5Cl_100p_run3": ("f3f04263709a27c2bc587c19d301ae39fbfd4d061f00595d998e240cb8c9709a", 841422),
    "Li6PS5Cl_100p_run4": ("f06be1c15797ceef134f044c468eda08a8462a0dc8e79a9c2c3522f79a564226", 841125),
    "Li6PS5Cl_100p_run5": ("a66cca8e08a034c870d3ce6c3ddc46c2380a6490a73d48bdb67d3ae843284015", 421259),
    "Li6PS5Cl_100p_run6": ("c0760787fa528afd1ff7e1b0b56515d1ec75eeb4ff2c8d13f5296ca3e2abd1bf", 421198),
    "Li6PS5I_0p_run1": ("f8327a7ff5ca2290b91f830d868defdac8c41df2f77b817dd855543c62bd4353", 837809),
    "Li6PS5I_0p_run2": ("bdd5a6754b4f86ca578db79b9bd75f8dce24e0406fd7093a8284de1b1792cf69", 837499),
    "Li6PS5I_0p_run3": ("a2fff2c7abca135cfdf2c7dc26f21dba20d4e7e0b26bde40fc7ddf1354445e98", 837903),
    "Li6PS5I_0p_run4": ("964547079960ba24024d90c40627ae73dd80a5ffa49accf5f04e21500a7c91fd", 838141),
    "Li6PS5I_0p_run5": ("975c87df4b8cf69b491782ade31ad3d44c04e0769c3b4b94f9376f4ae4ce50c0", 419661),
    "Li6PS5I_0p_run6": ("22c68154d0b7efd22821857283a78ce490117ed239f55da5a302d72564fe1121", 419582),
    "Li6PS5I_50p_run1": ("22ec30f4380ab545d68fd5b7e527e66877b12458268eb301b36d45896be13ba0", 601260),
    "Li6PS5I_50p_run2": ("088dae8cc61d4d87b6e77165b336f90055b2f409052e5a4b5ba804d3e2c29671", 601376),
    "Li6PS5I_50p_run3": ("d4e2148d4ed103e1ac8bed669677e49c254e1ae08e1369b06640a4dfb780fa1b", 601246),
    "Li6PS5I_50p_run4": ("a4dc66f576941a75bd1b269f659ea9c3ce14ac9eb2a58e89204e6e445d1b10e2", 601292),
    "Li6PS5I_50p_run5": ("88edd170c271744d1f569cd85adc9e0d838f5d52a00484afafafb8a6b494cc3b", 601231),
    "Li6PS5I_50p_run6": ("e865065ef883391abc83b13b5859627e5a363fa45246f66720a959511082ad19", 601306),
    "Li6PS5I_50p_run7": ("aa6310ece6b6242ecc189bb1285cc86381ff76198fe6195c4b4e851576e6af9d", 601252),
    "Li6PS5I_100p_run1": ("e5082c14dcba187dc0f2677d5e189cc8c69b0aef26c9c314efdf666172e06381", 600375),
    "Li6PS5I_100p_run2": ("d04112f14750b5bbf8a4427670bbb161f60c215b916ba0930c4b5cd5a1ccfffc", 600044),
    "Li6PS5I_100p_run3": ("cc85f40b8ddd07a79337c2dcc98577d542ba9f851e2c149ef20e628cb3d8977f", 600334),
    "Li6PS5I_100p_run4": ("a79e9d92badacf3165559089c9eda53dc95ada53863b2932a216b7d797d6a201", 600162),
    "Li6PS5I_100p_run5": ("0d026460c99bc79bd5eede2253d81d31af63b3bcc989b96c9da96eb4fd83172e", 600540),
    "Li6PS5I_100p_run6": ("269eec91b3e349a99908b23610894d0e0e18e325b55684f109bece42cdc0e98d", 600294),
    "Li6PS5I_100p_run7": ("60314dda4560a683595e91da15d1e99084ac3c8640504bb6d2f7bfdbf6ed5900", 600349),
}

_TIOF2_CIF = {   # 4x4x4 GA structures relaxed with CP2K
    1: ("ba4eae2b93de740135b3c09c6a93786327a24f167f0b6b232d7dd5265316f46a", 12834),
    2: ("98e6e823b6186e811b98d66d833487aa5e9f7ec45c1dba19b854e05f1a2d02fb", 12834),
    3: ("fe6b89a6fce2ede159bb209b9c4d6800f7bdbd55812603840fc46499b03ff2c6", 12834),
    4: ("9c65491a2d451d7451f56b37e877e4f5677f7208b21749f1b5198eba24272619", 12834),
}

_AB_HILLS = (
    ("d853a77ebee2dc13f778e15bf9e165e6e8b76ec67ef2ea2a3d722989c3859afa", 487360),
    ("4514f9c0f1a15bb1557369e83b8b2aa1abbd25c3b68b4830af84677c70064c75", 225353),
    ("b897259b2cfde0aa123d2a2dc5d9a1e4bcfa9cdc05204fd0b5d9c96d7b9718e7", 612093),
    ("e8af1f7860b36d286813d8664c31d49025cad13c04d4002987132e8c4905f9db", 235733),
)

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
        citation="J. L. A. Gardner, Z. F. Beaulieu, V. L. Deringer, Digital Discovery (2023)",
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
    Paper(
        name="argyrodite-disorder",
        title="Li6PS5X (X = Cl, I) argyrodites with 0, 50 and 100 % anion site disorder, AIMD at 500 K",
        citation="B. J. Morgan, Chem. Mater. 33, 2004 (2021)",
        doi="10.1021/acs.chemmater.0c03738",
        source="https://github.com/bjmorgan/data_argyrodite_disorder",
        license="MIT (figures CC BY-SA 4.0)",
        description="VASP XDATCAR.gz, 416 atoms, 700 frames 100 fs apart per composition and disorder, "
                    "stored as 5-7 consecutive segments (run1, run2, ...) that a wildcard joins",
        reproduces="msd.pdf, Li MSD against anion disorder for Li6PS5I and Li6PS5Cl "
                   "(MeanSquaredDisplacement on a wildcard trajectory)",
        script="argyrodite_disorder.py",
        files=tuple(RemoteFile(f"{key}.XDATCAR.gz", _github(*_ARGY, "data/{}/{}/{}/actual_XDATCAR.gz".format(*key.split("_"))),
                               sha, size) for key, (sha, size) in _ARGYRODITE.items()),
    ),
    Paper(
        name="tiof2",
        title="Correlated anion disorder in cubic TiOF2: relaxed 4x4x4 and GA-optimized 6x6x6 structures",
        citation="B. J. Morgan et al., J. Am. Chem. Soc. (2024)",
        doi="10.1021/jacs.4c06304",
        source="https://github.com/bjmorgan/data_TiOF2",
        license="CC BY 4.0 (data), MIT (code)",
        description="four CP2K-relaxed 4x4x4 cells (CIF, 256 atoms) and the 6x6x6 genetic-algorithm "
                    "anion arrangement (VASP POSCAR, 864 atoms)",
        reproduces="the Ti-F/Ti-O bond-length table (CreateBonds), the TiO(6-n)Fn octahedra "
                   "distribution against random and NbO2F-type order, and partial RDFs (CoordinationAnalysis)",
        script="tiof2_octahedra.py",
        files=(RemoteFile("6x6x6_opt.vasp", _github(*_TIOF2, "figure_plotting/Data/6x6x6_opt.vasp"),
                          "264ce37905a4af3a48d0d986d4706406d3eec663844bb75840cbf5354a925217", 59212),
               *(RemoteFile(f"config_{i}.cif",
                            _github(*_TIOF2, f"structure_prediction_dataset/vasp_4x4x4_from_ga/config_{i}/cubic/cp2k_opt/cp2k_opt.cif"),
                            sha, size) for i, (sha, size) in _TIOF2_CIF.items())),
    ),
    Paper(
        name="ammonia-borane-metad",
        title="Ammonia borane dehydrogenation, multiple-walker well-tempered metadynamics (PLUMED-NEST 19.082)",
        citation="V. Rizzi, D. Polino, E. Sicilia, N. Russo, M. Parrinello, Angew. Chem. Int. Ed. (2019)",
        doi="10.1002/anie.201900134",
        source="https://github.com/gilgalad33/Ammonia-Borane-Dehydrogenation",
        license="none stated (downloaded from the authors' repository, not redistributed)",
        description="PLUMED HILLS of 4 walkers (2 CVs: B-N intermolecular and B-H bridge coordination, "
                    "kJ/mol) and the authors' plumed sum_hills fes.dat",
        reproduces="the AB -> AaDB free energy surface, compared point by point with the authors' "
                   "sum_hills output (fes_from_hills of the ewald-plumed plugin)",
        script="ammonia_borane_fes.py",
        files=(*(RemoteFile(f"HILLS.{i}", _github(*_AB, f"Ammonia_borane/1.AB_to_AaDB/HILLS.{i}"), sha, size)
                 for i, (sha, size) in enumerate(_AB_HILLS)),
               RemoteFile("fes.dat", _github(*_AB, "Ammonia_borane/1.AB_to_AaDB/fes.dat"),
                          "29d978e0a5f034f3101f0f36ef1c236e7463a16f6a9b5cb45caf3179ce970594", 3071649)),
    ),
    Paper(
        name="alanine-dipeptide-metad",
        title="Alanine dipeptide in water, well-tempered metadynamics on phi/psi (PLUMED-NEST 20.023)",
        citation="D. Trapl, V. Spiwok, The R Journal 14(3) (2022)",
        doi="10.32614/RJ-2022-057",
        source="https://github.com/spiwokv/metadynminerdata",
        license="none stated (downloaded from the authors' repository, not redistributed)",
        description="PLUMED HILLS (30,000 hills on phi, psi; kJ/mol, 300 K, bias factor 10) and plumed.dat",
        reproduces="the Ramachandran free energy surface and its minima (fes_from_hills of the ewald-plumed plugin)",
        script="alanine_dipeptide_fes.py",
        files=(RemoteFile("HILLS", _github(*_ALA, "2D/HILLS"),
                          "b5729a8968d8978ce926ae6773551de7266ebbc2047c22d367ce317c8fa63573", 4860156),
               RemoteFile("plumed.dat", _github(*_ALA, "2D/plumed.dat"),
                          "dcdc7b977820770ff166576671cf04722c87f91c25ed50227f930c654a041bdc", 206)),
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
