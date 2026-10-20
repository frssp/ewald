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

from .._tls import ssl_context


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
_NBO2F = ("bjmorgan/data_nbo2f_chirality", "2781ddde04020d618eecdb0616b409c24545ec5b")
_PMG = ("materialsproject/pymatgen", "c6507866179a6c37ee7d0df6cbb79896a2ebba52")
_PHONOPY = ("phonopy/phonopy", "c145fccf0d440bef14e6e3544218a7d2b31a37ae")

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

_NBO2F_OUTCAR = {   # 3x3x3 config number: (sha256, size)
    0: ("51f1db8a2a89bf911ead9f4fa49765ea588eaf69eb789eda3e463d7c0cc15e83", 209104),
    1: ("5f84c60deb9d8ee72d080c34240b193c4ec7f684af80bfafd19d2be70f8483c2", 209104),
    2: ("c2358f70f3c1327ae14acd970679836a63d6d329493ec81c7106ab0df68d01fe", 209104),
    3: ("c2ca6792eca1351f76b3fc1eac544f2770a82c28f4a5294fc697e2bd3e416017", 209104),
    4: ("52f421c3370c5e138f714f44aa14b221a8e668f5448769412365c931600ad68f", 209104),
    5: ("cfb9d24f2be7ffc5e90c6c3a8c3bffa7272fffbc67369499be114789af0da4bd", 209104),
    6: ("699630668ed4df4dec6b265dec09bea2aaa306fcf5f9a20ba355545c01264a75", 211072),
    7: ("cba9bd1115b374385411b3ebbcfb36bb25b24d39821c065262b5d73d76e7e0c2", 209104),
    8: ("508d72345e7c91f103f9b86c8b70b2eb5dae21f572f71249461478e4aff380f4", 209104),
    9: ("06b95b410b3823cea1620595a2d5455e18220a9765dc250d63a2bcab1f1d06e9", 209104),
    10: ("b6e38752367411b8616250d23ffe2755e7fa0d73b8b7c601914cfd97731ce303", 213048),
    11: ("dcdf918e0547377069d8da18da3b27852c5f3a21700ed965bbbd16e9c8af3539", 209104),
    12: ("938065c00bdbd9e53f85034f1dda9f25772bafcd828174820749fc6ca785e4df", 209104),
    13: ("4e8a5d268b2e919f1966d5518217daac946a7875092ed7c71e8971c56a6c5fb9", 209104),
    14: ("acbe17e8ee52c954f415d705ffa843a619bccb598d51e63d17731dff7a42af4b", 209104),
    15: ("6b96df859c381e0b3999dde34a65da0c89537f5773a821176d2c55474d92b218", 209104),
    16: ("2407ac416a8f99768f7704f4bbc98aa060f65603742439d585548e0287653833", 209104),
    17: ("b093a7888f43366a4040a94c02a9b0e273fd6db44ba1633949328db825600f42", 209104),
    18: ("da4fe0bce14a9ca8ab716db3ca1c0447279b42e2caaade36b6e512b925da653f", 209104),
    19: ("1dfa485a3bcb805042db95bf598c8e9c6d6a3213a797d3cea0d211f51818ddc7", 209104),
    20: ("5ec68191e4680a99e308cea2bad3c68fa9670a0fc2a34c494e8d7dd3cef15db6", 209104),
    21: ("2e2f42968fbbaa8cab838521784a1b5791ca5b56f69fe22688cab6a2d2b4f2f6", 209104),
    22: ("7e6f559529f3126c898c41581b737df5f29c41670c10456c4ef5485ad24e830e", 209104),
    23: ("8e1f32404261485b2cae65d20446a3b950991cfb0f9b0422b7e720dcb130e9fb", 209104),
    24: ("3fd9cd7fc5ea4c5d3dcbb248dcbe61dcac0af867ab3e0b25a921642de351d12c", 211072),
    25: ("e2982bd212fa15cf50710db8d6fc4cb6384ff5e459fa41460936ff599219fba9", 209104),
    26: ("7f97155af42312843aec1d832bccc47c1b7e388aa82ae2b8b4e3202802b4e8ed", 209104),
    27: ("ea356481f3cb2853a2ae02d232327db21816a81c69006289d46e1bae0dc917bd", 209104),
    28: ("6216ce2432300fa0394a64204c10d96db99bd9b837a12265a3554959f8dbdc33", 209104),
    29: ("be19c80d08cc55e52bdee00d6e7c67ecacda7a203d60590bf12a376ca843fac6", 209104),
    30: ("0375b58b12600fbe12b8e7be0a09184756b49d2af0038559abf246c367e28005", 209104),
    31: ("18a1135eaafa587230e64f328459b61a31813e95f095038a1ed2d37c01970156", 209104),
    32: ("aa4834277bb127d4124037306c6d2c02e4af0fe86012a8043d16945f0382e442", 211072),
    33: ("e85e6486d1e5b7a60de977ab3aea7cbf3f1dc5be94978e154784b7e70e20ddba", 209104),
    34: ("2738e546e7b6c1faacb05f67290c0331544ed985a46fdf7731ea02b7fc0b6a83", 209104),
    35: ("4752b911ce5882cc0896ac4da57b5bdf0f3fc79d75c7c0edb39376f50df0f7d2", 209104),
    36: ("051a32b4fe77ddca36f89d1918bfc036f78ab8c31e165b3757832973a5503e1c", 209104),
    37: ("63463838a0e7f1d509f368318188671dbfb8c1fea376ce4cfdd3e964a506f7a4", 211072),
    38: ("18cc05dfd0689e36c53d2dc00a7a149a8af8498a2cfd22f9d36615e65b0d145e", 209104),
    39: ("bf10b98185e76030b7688cb8d1274e541559f6e32a5fc3dadaf4dc5647e26a97", 209104),
    40: ("1e938adf29ea1d35d391eaa45f4ed61f644893d844388a2f136c7fe22763ed3b", 209104),
    41: ("448413dba2feb03ccc13fe5155d76e828fdfd4481a272c1e1f26551bd89e1821", 209104),
    42: ("e5e111e6441b5916e869b0f45f179a7624231e557ee22a7a21c29cfec6eeaaa6", 209104),
    43: ("2d22e395e1b210cfbd7a2a307b8df56cf2d72b83cd58d9b525e7aa57c5ac7f83", 213048),
    44: ("3801aa00e8ccba169905c2cb329758ad42d8c0cac0dc583df822b11de8858959", 209104),
    45: ("d0502333928670b3cba900b5213541952045a5fde1c788b3ffd3f5aac5b2a8c2", 209104),
    46: ("7cb4bb5f39fa1e10a5521abff6ccd2007b065713ef73ab4fc7618aae787218e1", 209104),
    47: ("35c28143a2f3c48b58b9c13a2c8472d0c0eb8b551ef4f6549d6939b064a1adc9", 209104),
    48: ("a970955e29468bd45d6c6fa76078d14f82cfa809f755d91727f4f44c621cdffc", 209104),
    49: ("df9cfa1b7db95a49d878f6166577c1f38928729cd5b2ff12228dbb626cb00c98", 209104),
    50: ("31879f08c3055ef70fdf95c95067995a2c4393ccadbe7500379ddb5ae3a0935b", 209104),
    51: ("29d4e57461e5dd902888653e895dcc622dc4cf62a3133a224a3052f8166e3bb5", 209104),
    52: ("341a2927bea31166428e74d8aa2fbbf7ef9ab65ef6f8cd70079d78e903c10247", 209104),
    53: ("27e0817094bcd809fbdd4b678a17c9c1e91d9ae121a95e3830675a93f3cf8cef", 211072),
    54: ("7c0f81e6b43ea0f003ff21c4d8b4b31e411d2b874b1884a1a70e003ac6638cb3", 209104),
    55: ("1d177c6c43af2d488ff53ae963811b3f84136b3250cb7aa5c04e9b85e4bc8e3f", 209104),
    56: ("0b9e7eed2d6a304c0884909e6e19110795f5f0bc5110dc0096ea22fa0f63a5d4", 209104),
    57: ("8d77440062fe5b67f403651ceedff9f13af199d6a03b917e2d0d8eaddcd717b9", 211072),
    58: ("3e57952e4225b7ea0d026979767526aad2347db06e8ef240d70bb9297908a528", 209104),
    59: ("c9145996d1fbb7e7de06b2100b6250356f46f380f2807541d0edc96995aa4fe8", 209104),
    60: ("c66824f95b49e96a0ef03a3a6ea9d0940368e1ecac9c5423e0ced36c9b2bb417", 209104),
    61: ("4ca4981a95e69c43038a2a075f1158a7d5a4b7d93fe95e8425244306fe9fe6ba", 209104),
    62: ("3fdf852d512ead59db0a5986dd1a88379e65acacd98bb5d68ca30c313eba0942", 209104),
    63: ("05120f9d0ff8a9b36c919e0d43ee366331a67d318c42462fd781c83e6cc9e370", 207144),
    64: ("2327410e1345b4109e7fd8174d7e49f8522718f7c31d59a8d24f85df3bc8944e", 207144),
    65: ("8d8fee500f29142596a780370638d8dfb62dbc15bcdabd948b07266b61b2566e", 207144),
    66: ("2e5bdca7d8cabdf6c2cd300e65ad7b033fa9c108efb92701aaccf359977ef322", 207144),
    67: ("8f8f2a643fe88ce6af1593d596b28aa678e8b1dbc5e68e00033027169e2e5436", 207144),
    68: ("34c1372ee4276688bc0c66b121776d9304c19f6ecbc9c6ee8939fb6a30a9e345", 207144),
    69: ("b21ab5138d7464d23a2a094fd74d10485b1e9d07df0e964e71ae588e5248429e", 207144),
    70: ("59089f48636b0f31906ce037971b848d08d7597077b5df552bd34f7388c9543a", 207144),
    71: ("66781b0863df99a97a2241086528b9bb87a77a63c57dacb1fcf399c19245c1b0", 207144),
    72: ("d879a6d5c3238b7ccde5d4b0fe61706531172ad8e91a549f83ac60cd4741cdc1", 207144),
    73: ("4767e73bb87a072673622cf91bae00b2915879ee3a93ee3fe5fdcf8cee4eb9e9", 207144),
    74: ("8406882c8f47326c696422b013bef385f79f29a93e872254e132ad5ff082a32f", 209104),
    75: ("237c1b157e9ed66e0565e968363408a3413c44ddabf487777a326a909e04c753", 209104),
    76: ("005de78f2d4216f69d84199cf068334289209440a89bd14c3790c4a2ef16c1e9", 209104),
    77: ("6d58a7c15612ee272494b630e0cd0864018dbbfd7a72801b31bc7fa7fa4ca1e9", 209104),
    78: ("fd7042a882e025b27959dc7fd72bcc5fe8b318e0d86f2dd869c2afca1bcc5972", 209104),
    79: ("c8fb489d10306369b7c66a8166ace75e398ef46f2f8e893c572a845e650387c1", 209104),
    80: ("093b9fdccf80436cbfcdcfa1f7529f009ae7cbde266d9e6f025cecb62b511262", 223048),
    81: ("a0844af1e0a0abb12b62f3c1dea84b9d74601fbac5ea4f2607c614a2b2ec9f1c", 207144),
    82: ("c7c917897b27b1fc83ae1ea2820da6ebde76351d773a9e9bf07f1dde0949e6cf", 209104),
    83: ("22a8e119ee78618dcb9a8bb94ca098eef483856280c08318b33c253626853017", 207144),
    84: ("cdb2184e0a5a411cccf693700d3353f411f50580a97c7db29f3e0d0cae22e197", 207144),
    85: ("0ac4986547231f7173318bd3e5b7b662a23894d23887c757526c6ccbd49bda77", 209104),
    86: ("5d6ef98bb589d77ca15d78f859119e485c869d7d69cc9bdf26d5dc5c24562f83", 209104),
    87: ("152e6b8af7adbfb859431bfdb0239356517e6d1db87fdf001c0b8cfd7b26dfa9", 207144),
    88: ("50ccc571222177c767a54af9523fed0f56275d5072a66c74013305689a34fbaf", 207144),
    89: ("d211942589555136923b8161499bf6cc56efdd0d2fb7f60e7171ee2ceb11df3e", 166238),
    90: ("e6826e794ae774c4d80cdd9f95371fc713539be5701d390a60a914b619af42db", 166238),
    91: ("fd985cf028eb164e4be74e3489b2740da717d94f7865e35de099cd7bb4d4b9a6", 166238),
}
_FE_NEB = {   # image folder/file: (sha256, size); VTST layout, endpoints with OUTCAR and POSCAR
    "00/OUTCAR": ("6eb7b35b42f3699866312ff566b0acd8558d470e45b84a98c6a395d03d5551ee", 3955168),
    "00/POSCAR": ("0bdd5e6f9c682ece62efabbd292500121084291200d3ba76ee59e93d43593650", 1038),
    "01/CONTCAR": ("880d4284aa880f005d54bf0ee63f8b1c5f68585677a738adc554ce39f495fdcb", 3660),
    "01/OUTCAR": ("1cf7e47231d14b91515a70a046426199ca092608e3702f57183fbff848dede83", 8684066),
    "02/CONTCAR": ("52691be1853f7ebbb4efe3381f8d614ca91168b768992f98a5ac9a17fa5bb2b1", 3660),
    "02/OUTCAR": ("0dab8255e3932d102aaaa6f35f0664d6d7019f25a8c2c58f5dbc0ec8d9188d9c", 8610614),
    "03/CONTCAR": ("d3ca866806d1bfaec947c630a271ba611cf761f134148be77f60a8186ca45213", 3660),
    "03/OUTCAR": ("bdf04813f3ea4180f25882d4e008295d7bca11e7e29f35fc64e9b59079983a1a", 8684064),
    "04/OUTCAR": ("a221b302065d5e2b09dc350e18dd0de90785dc4f7b41f377c47f46965ba2534b", 3951218),
    "04/POSCAR": ("5c576e766d3f453d2ca44779ece03d95a2ed50dc637883de3d470502e55315e8", 1872),
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
    Paper(
        name="nbo2f-anion-order",
        title="NbO2F anion orderings: 92 VASP relaxations of a 3x3x3 supercell",
        citation="B. J. Morgan, \"Structural Chirality from Short-Range Order in Heteroanionic Materials\", "
                 "arXiv:2608.04841 (2026)",
        doi="10.48550/arXiv.2608.04841",
        source="https://github.com/bjmorgan/data_nbo2f_chirality",
        license="CC BY 4.0 (data), MIT (code)",
        description="92 VASP OUTCARs (PBEsol, full relaxation, ISIF=3) of Nb27 O54 F27 with random, "
                    "period-three and the twelve enumerated cis + period-three anion orderings",
        reproduces="the DFT ground state: the enantiomer pair P3_121 / P3_221 (orbit 11), degenerate and "
                   "lowest of all 92 (load_runs, check_run, SymmetryAnalysis), and cis-NbO4F2 octahedra",
        script="nbo2f_anion_order.py",
        files=tuple(RemoteFile(f"config_{n:03d}/OUTCAR", _github(*_NBO2F, f"data/dft/3x3x3/config_{n:03d}/OUTCAR"),
                               sha, size) for n, (sha, size) in _NBO2F_OUTCAR.items()),
    ),
    Paper(
        name="fe-vacancy-neb",
        title="Vacancy migration in fcc Fe: a VASP climbing-image NEB (pymatgen test data)",
        citation="S. P. Ong et al., Comput. Mater. Sci. 68, 314 (2013) (pymatgen; NEBAnalysis test data)",
        doi="10.1016/j.commatsci.2012.10.028",
        source="https://github.com/materialsproject/pymatgen",
        license="MIT",
        description="VTST NEB folder 00-04: Fe31 (32-site fcc cell with one vacancy), three images with OUTCAR "
                    "and CONTCAR, endpoints with OUTCAR and POSCAR",
        reproduces="the 325 meV barrier and the saddle at half the path that pymatgen's NEBAnalysis gets "
                   "(NebEnergyProfile on the folder, OverlayFrames)",
        script="fe_vacancy_neb.py",
        files=tuple(RemoteFile(name, _github(*_PMG, f"test-files/io/vasp/fixtures/neb_analysis/neb1/neb/{name}"),
                               sha, size) for name, (sha, size) in _FE_NEB.items()),
    ),
    Paper(
        name="mgb2-phonons",
        title="MgB2 phonons from VASP finite displacements (phonopy example)",
        citation="A. Togo, J. Phys. Soc. Jpn. 92, 012001 (2023); band path of K.-P. Bohnen, R. Heid, "
                 "B. Renker, Phys. Rev. Lett. 86, 5771 (2001)",
        doi="10.7566/JPSJ.92.012001",
        source="https://github.com/phonopy/phonopy",
        license="BSD-3-Clause",
        description="phonopy_disp.yaml and FORCE_SETS of a 3x3x2 MgB2 supercell (VASP, 500 eV, 18x18x18 k-points "
                    "for the unit cell)",
        reproduces="the band structure along Bohnen's path and the Gamma modes E1u, A2u, E2g, B1g, with the "
                   "E2g boron bond-stretching mode animated (PhononMode of the ewald-phonopy plugin)",
        script="mgb2_phonons.py",
        files=(RemoteFile("phonopy_disp.yaml", _github(*_PHONOPY, "example/MgB2/phonopy_disp.yaml"),
                          "780fb6c8455096960eae4f8ab436c30e5d554743fdaf54b3472533fcfa878ce8", 9982),
               RemoteFile("FORCE_SETS", _github(*_PHONOPY, "example/MgB2/FORCE_SETS"),
                          "5400dd39e2079e078ce4f7728bcfb3689c70e272fdbf1e9936e68cad7ecaf6df", 5322)),
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
    with urllib.request.urlopen(url, timeout=timeout, context=ssl_context()) as resp, open(tmp, "wb") as out:
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
        path = folder / f.name                 # a name may hold a sub folder (NEB images 00/, 01/, ...)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.stat().st_size == f.size and _sha256(path) == f.sha256:
            continue
        if not quiet:
            print(f"downloading {f.name} ({f.size / 1e6:.1f} MB) from {paper.source}", file=sys.stderr)
        _download(f.url, path)
        if _sha256(path) != f.sha256:
            path.unlink()
            raise ValueError(f"{f.url}: checksum mismatch (the file changed upstream?)")
    return folder
