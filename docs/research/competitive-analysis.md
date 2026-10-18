# 경쟁 제품 분석 및 재사용 가능 오픈소스 조사

2026-10-18 기준 · 원본: [Claude Doc](https://claude.ai/code/artifact/614c2459-62b4-489c-9f48-bb2f22ad1128)

## 요약

가장 큰 발견은 OVITO 소스가 GPL 전용이 아니라 **GPLv3 또는 MIT 중 선택**이라는 점입니다. Repo 루트의 [LICENSE.txt](https://gitlab.com/stuko/ovito/-/raw/master/LICENSE.txt)와 각 파일 header에 "either under the GPL or, at your option, under the MIT License"라고 명시되어 있어, CNA, PTM, Voronoi, Wigner-Seitz, grain segmentation 같은 검증된 C++ 알고리즘을 MIT 고지만 남기고 우리 코드로 port할 수 있습니다.

다만 공개 repo(`gitlab.com/stuko/ovito`, 2026-09-12 최신 commit)에는 DXA, Python binding(Python script modifier), OpenGL/OSPRay/Tachyon renderer 코드가 없습니다. 이 셋이 OVITO GmbH가 닫아 둔 부분이고, 우리가 무료로 메워야 할 핵심 gap입니다.

프로젝트 plan에 주는 영향은 세 가지입니다.

1. "OVITO는 GPL이라 코드를 가져오지 않는다"는 전제를 바꿔, 분석 modifier는 freud와 함께 OVITO MIT 코드를 reference/port 대상으로 씁니다. 프로젝트 license는 MIT/BSD로 유지할 수 있습니다.
2. DXA는 여전히 공개 소스가 없으므로 범위 밖이 맞습니다. 대신 MIT로 배포되는 `ovito` pip module을 optional backend로 호출하는 방법을 검토할 만합니다(Pro 기능 사용 조건은 확인 필요).
3. GUI 구조는 tomviz(Qt + VTK + 편집 가능한 Python operator, BSD-3)가 가장 가까운 open-source 선례라 architecture reference로 씁니다.

## 경쟁 제품 비교

재료 시뮬레이션(VASP/LAMMPS) + 분석 + Python을 한 번에 갖춘 도구는 OVITO뿐이고, 나머지는 분야(생체분자, 결정구조, volume)나 형태(web, notebook)가 다릅니다. License는 OVITO, VMD, tomviz, freud, MolecularNodes, pygfx, VIAMD/mdlib만 원문을 확인했고, 나머지는 기억 기반의 approximate 값입니다.

| 제품 | 형태 / stack | License | 강점 | 우리 입장에서의 약점 |
| --- | --- | --- | --- | --- |
| [OVITO Basic](https://gitlab.com/stuko/ovito) | Desktop, C++/Qt | GPLv3 or MIT (확인) | Modifier pipeline, CNA/PTM/Voronoi/WS/grain, LAMMPS·VASP reader | GUI Python, DXA, renderer가 공개 소스에 없음 |
| [OVITO Pro](https://ovito.org/manual/reference/pipelines/modifiers/index.html) | Desktop + `ovito` pip | Proprietary (pip module은 MIT) | Python script modifier, time series, spatial binning, OSPRay/VisRTX | 유료, closed |
| [VMD](https://tcbg.illinois.edu/Research/vmd/current/LICENSE.html) | Desktop, C++/Tcl/Python | Academic license (확인) | 대용량 MD, Tachyon/OptiX 렌더, plugin reader | 재료 분석(CNA/PTM) 없음, 코드 재배포 제한 |
| VESTA | Desktop | Freeware, closed | 결정구조, charge density(CHGCAR) isosurface, 논문 figure 표준 | Trajectory·scripting 없음 |
| ParaView / VisIt | Desktop, VTK/Qt | BSD-3 | 범용 과학 시각화, pvpython, 병렬 렌더 | 원자 전용 UX 없음 |
| [tomviz](https://github.com/OpenChemistry/tomviz) | Desktop, Qt6 + ParaView + VTK + ITK | BSD-3 (확인) | GUI에서 편집하는 Python operator pipeline | Tomography volume 전용, 원자 구조 없음 |
| Jmol / JSmol | Desktop + web, Java/JS | LGPL | CIF/결정, scripting 언어 | 성능, Python 없음 |
| nglview / NGL, Mol\*, 3Dmol.js | Jupyter/web, WebGL | MIT / MIT / BSD | Notebook 안 trajectory | 재료 분석, publication 렌더 약함 |
| PyMOL (open-source) | Desktop, C++/Python | BSD-like | Python API, 렌더 품질 | 단백질 중심, PBC 약함 |
| ChimeraX | Desktop, Python/Qt | Non-commercial | Python 내장, 현대적 렌더 | 재배포 제한, 재료 분석 없음 |
| Avogadro 2 | Desktop, Qt/VTK | BSD-3 | 분자 편집, Python plugin | 대용량 trajectory 약함 |
| ASE gui, crystal-toolkit, chemiscope | Tk / Dash / web | LGPL / MIT / BSD | 가벼움, Python native | 분석·렌더 기능 제한 |
| Blender + [MolecularNodes](https://github.com/BradyAJohnston/MolecularNodes) / Beautiful Atoms | Blender add-on | GPL-3 (확인, MN) | 최고 수준 ray-traced figure·animation | 분석 없음, GPL |

공유해 주신 링크에서 추가로 본 도구입니다.

| 제품 | 형태 / stack | License | 강점 | 우리 입장에서의 약점 |
| --- | --- | --- | --- | --- |
| [VIAMD](https://github.com/scanberg/viamd) + [mdlib](https://github.com/scanberg/mdlib) | Desktop, C/C++ + OpenGL (+ Metal compute), Dear ImGui | MIT (확인, 둘 다) | GPU 중심 렌더로 대용량 trajectory를 부드럽게 표시, frame 전체에 걸친 선택/속성 scripting 언어, LAMMPS·XYZ·XTC·TRR·GRO·PDB·mmCIF·cube reader | 생체분자 MD 중심, VASP reader 없음, 재료 분석(CNA/PTM) 없음, Python 대신 자체 언어 |
| [JLmol](https://jlmol.com) ([GitHub](https://github.com/fkfest/jlmol)) | Desktop + web, Electron + JSmol | LGPL-2.1 (홈페이지 표기) | 깔끔한 UX, PubChem 검색, xtb 계산 연동, MO 표시 | 분자 중심, VASP/LAMMPS·trajectory·Python 언급 없음. "계산 코드와 붙은 viewer" UX는 참고할 만함 |
| XCrySDen | Desktop, Tcl/Tk + OpenGL | GPL (approximate) | 결정 + isosurface, Fermi surface | 오래된 UI, 분석 없음 |
| AtomEye, GDIS, Aten | Desktop | Free / GPL (approximate) | 가벼운 대용량 viewer(AtomEye) | 사실상 유지보수 중단 |
| Atomsk, Packmol, Moltemplate, AtomMan | CLI / Python, 구조 생성 | GPL-3 / MIT / MIT / BSD (approximate) | Dislocation·polycrystal·혼합물 초기 구조 | Viewer 아님. "구조 만들기" 메뉴에서 외부 호출 대상 |

출처: [Atomsk links](https://atomsk.univ-lille.fr/links.php).

포지셔닝으로 보면 "OVITO Basic의 분석 + OVITO Pro의 Python + VESTA의 volumetric + Blender급 figure"를 모두 MIT/BSD로 묶은 제품은 현재 없습니다.

## OVITO Pro 전용 기능과 메울 방법

현재 [OVITO 매뉴얼](https://ovito.org/manual/reference/pipelines/modifiers/index.html)은 modifier 17개를 Pro 전용으로 표시하고, [rendering 항목](https://ovito.org/manual/reference/index.html)에서는 OSPRay, Tachyon, VisRTX renderer와 AI Agent를 Pro로 표시합니다. 대부분 numpy/scipy/freud 수준의 간단한 계산이라, 우리 쪽에서는 가장 많이 쓰는 것부터 싸게 채울 수 있습니다.

| Pro 기능 | 난이도 | 구현 방법 |
| --- | --- | --- |
| Python script modifier, Python file reader, Python code generator | 중 | 핵심 차별점. Pipeline이 원래 Python이라 함수 등록 + 내장 qtconsole로 구현. Code generator는 pipeline 상태를 Python source로 직렬화 |
| Time series, Time averaging, Reduce property, Difference between frames | 하 | Frame loop + numpy. Lazy frame cache 위에 얹기 |
| Spatial binning, Structure factor | 하 | numpy histogramdd, freud `StaticStructureFactor` |
| Bond angle / Bond length distribution, Find rings | 하~중 | Neighbor list + numpy, ring은 networkx 또는 [matscipy](https://github.com/libAtoms/matscipy) ring statistics |
| Bond order (Steinhardt), Calculate local entropy | 하 | freud `Steinhardt`, local entropy는 논문 수식 직접 구현 |
| Bader charge integration | 중 | VASP CHGCAR + 외부 `bader` 또는 pybader 호출 |
| Identify fcc planar faults | 하 | PTM 결과 후처리(OVITO MIT 코드의 PTM 구현과 함께) |
| Shrink-wrap box, Color by type, Render LAMMPS regions | 하 | 몇 십 줄 |
| OSPRay / VisRTX / Tachyon renderer | 중 | VTK의 OSPRay·OptiX(ANARI) pass 사용 또는 Blender export |
| DXA (Pro는 아니지만 공개 소스 없음) | 상 | `ovito` pip module을 optional backend로 호출, 자체 구현은 보류 |

차별화의 핵심은 Python script modifier를 GUI에서 무료로 제공하는 것과, VESTA가 하는 CHGCAR/PARCHG isosurface를 같은 화면에서 처리하는 것입니다. OVITO도 CHGCAR isosurface를 지원하지만, VESTA식 결정구조 표현(polyhedra, lattice plane, 단면 contour)과 함께 쓰기에는 약합니다(기억 기반, 확인 필요).

## 가져다 쓸 수 있는 컴포넌트

MIT/BSD/Apache/LGPL은 MIT 프로젝트에서 바로 쓸 수 있고, GPL은 optional plugin으로만 분리합니다. "확인"이 붙지 않은 license는 approximate입니다.

### OVITO 공개 소스에서 port할 수 있는 것 (MIT 선택, 확인)

[gitlab.com/stuko/ovito](https://gitlab.com/stuko/ovito) `src/` 아래 약 2,100개 파일을 확인했습니다. 모두 OVITO 자체 framework(Qt, 자체 객체 시스템)에 묶여 있어 drop-in은 안 되고, 알고리즘 부분을 떼어내 pybind11/numba로 옮기는 방식입니다.

| 위치 | 내용 | 쓰임 |
| --- | --- | --- |
| `particles/modifier/analysis/` | Ackland-Jones, centrosymmetry, Chill+, cluster, CNA, diamond, displacements, PTM, RDF, atomic strain, surface, Voronoi, Wigner-Seitz | 3단계 분석 modifier의 정답 구현이자 test oracle |
| `crystalanalysis/modifier/` | Grain segmentation, elastic strain, structure analysis | Grain 분석 |
| `vorotop/`, `correlation/`, `mesh/surface/` | VoroTop, spatial correlation, construct surface mesh(alpha shape) | 고급 modifier |
| `particles/import/`, `export/` | LAMMPS dump/data, POSCAR/XDATCAR/OUTCAR, extended XYZ 등 reader | ASE reader보다 빠른 대용량 dump parser의 reference |
| `3rdparty/ptm` | PTM C++ 원본 (MIT, 확인) | 그대로 vendoring 가능 |
| `3rdparty/voro++` | Voro++ (BSD-style, 확인) | Voronoi |

공개 소스에 없는 것: DXA, Python binding/script modifier, OpenGL·OSPRay·Tachyon renderer.

### Layer별 라이브러리

| Layer | 후보 | License | 메모 |
| --- | --- | --- | --- |
| I/O | ASE, pymatgen | LGPL-2.1+ / MIT | VASP 전반(CHGCAR, OUTCAR 포함) |
| I/O (대용량) | [chemfiles](https://github.com/chemfiles/chemfiles), MDTraj, GSD | BSD-3 / LGPL-2.1 / BSD-2 | LAMMPS dump, XTC, DCD 등 빠른 C++ reader |
| I/O | MDAnalysis | GPL-2+ → LGPL-3+ 전환 제안([2022](https://mdanalysis.org/2022/11/07/relicensing), 완료 여부 미확인) | 완료 전에는 optional로만 |
| 분석 | [freud](https://github.com/glotzerlab/freud) | BSD-3 (확인) | RDF, Steinhardt, Voronoi, cluster, environment matching, S(k) |
| 분석 | spglib, seekpath | BSD-3 / MIT | 대칭성, k-path |
| 분석 | matscipy, pyscal3 | LGPL-2.1 / GPL-3 | Ring statistics, Steinhardt 등. pyscal은 GPL이라 optional |
| 분석 | `ovito` pip module | MIT (확인) | DXA optional backend. Pro 기능의 pip 내 사용 조건은 확인 필요 |
| GPU 계산 | CuPy, PyTorch | MIT / BSD-3 | Neighbor list, RDF, MSD |
| 렌더 | VTK + PyVista | BSD-3 / MIT | Glyph instancing, OSPRay·ANARI pass, offscreen |
| 렌더 (대안) | [pygfx](https://github.com/pygfx/pygfx) (wgpu) | BSD-2 (확인) | 차세대 후보, 1.0 전이라 API 변동 |
| 렌더 (대안) | [mdlib](https://github.com/scanberg/mdlib) | MIT (확인) | GPU sphere impostor, SSAO, `md_lammps.c` reader |
| Ray tracing | OSPRay, Blender export | Apache-2 / GPL(Blender 외부 프로그램) | 논문 figure. Blender는 파일 export만 하면 license 영향 없음 |
| GUI | PySide6 | LGPL-3 | PyQt(GPL) 대신 |
| GUI | Qt Advanced Docking System, pyqtgraph | LGPL-2.1 / MIT | Dock layout, plot panel(RDF, histogram) |
| 내장 Python | qtconsole + ipykernel | BSD-3 | GUI 안 IPython console |
| Notebook | nglview, trame | MIT / Apache-2 | Jupyter에서 같은 scene 표시 |
| Architecture 참고 | [tomviz](https://github.com/OpenChemistry/tomviz), napari | BSD-3 / BSD-3 | Python operator pipeline UI, plugin 체계 |

## 라이선스 전략

프로젝트 본체는 MIT(또는 BSD-3)로 두는 것을 권합니다. OVITO 코드를 MIT 조건으로 port할 수 있고, freud·VTK·PyVista·CuPy·PTM·voro++가 모두 호환되며, 사용자 사내 배포나 상용 연동에도 제약이 없기 때문입니다.

1. OVITO에서 가져온 파일은 원본 copyright와 MIT notice를 유지하고 `third_party/ovito/`에 둡니다. 파일 header의 "GPL or MIT" 문구가 있으니 MIT 선택을 README에 명시합니다.
2. LGPL(ASE, PySide6, MDTraj)은 동적 import만 하면 문제가 없습니다. 번들 배포 시 사용자가 교체할 수 있게만 두면 됩니다.
3. GPL(MDAnalysis 현행, pyscal3, Atomsk, MolecularNodes)은 core가 import하지 않는 optional plugin 패키지로 분리합니다.
4. VMD, ChimeraX, VESTA 코드는 사용하지 않습니다. VMD는 재배포가 "1/10의 코드에서 1/2 이하 파생"으로 제한되고 상용은 별도 license가 필요합니다([VMD License](https://tcbg.illinois.edu/Research/vmd/current/LICENSE.html)). VMD plugin(molfile)은 UIUC Open Source License라 예외입니다.
5. `ovito` pip module은 MIT로 배포되지만([Licensing information](https://ovito.org/manual/licenses/index.html)), 바이너리 안의 Pro 기능을 무료로 써도 되는지는 문서에 명시가 없습니다. DXA처럼 Basic에 있는 기능만 optional로 호출하는 것이 안전합니다.

## 권장 스택과 다음 단계

기존 plan의 스택(PySide6 + PyVista/VTK, ASE reader, freud, CuPy)은 그대로 유효하고, 아래 네 가지만 더하면 됩니다.

1. **분석 modifier의 기준을 OVITO MIT 코드로.** CNA, PTM, Wigner-Seitz, grain segmentation은 freud에 없거나 정의가 다릅니다. OVITO 구현을 port하고 `ovito` pip 결과와 숫자를 비교하는 regression test를 둡니다.
2. **PTM과 voro++는 vendoring.** 둘 다 작은 C++ 코드라 pybind11로 바로 감쌀 수 있습니다.
3. **대용량 LAMMPS dump는 chemfiles 또는 자체 reader.** ASE reader는 수백만 atom dump에서 느리므로, OVITO의 dump parser를 reference로 numpy 기반 reader를 넣습니다.
4. **GUI 구조는 tomviz를 참고.** Qt6 + VTK 위에서 Python operator를 GUI에서 편집·저장하는 방식이 OVITO Pro의 Python script modifier와 거의 같습니다. 코드는 BSD-3라 필요한 부분을 가져와도 됩니다.

렌더링에서는 VIAMD/mdlib(MIT)가 가장 쓸모 있는 참고입니다. VTK glyph 방식이 수백만 atom에서 느려지면, mdlib의 GPU sphere impostor·SSAO 방식을 별도 OpenGL viewport로 port하거나 mdlib을 C library로 묶어 쓸 수 있습니다. `md_lammps.c`도 빠른 dump reader의 후보입니다.

DXA는 1차 범위에서 빼고 `ovito` pip optional backend로만 제공하는 것을 기본값으로 잡았습니다.

## Sources

- [OVITO source repository](https://gitlab.com/stuko/ovito) · [LICENSE.txt](https://gitlab.com/stuko/ovito/-/raw/master/LICENSE.txt)
- [OVITO licensing information](https://ovito.org/manual/licenses/index.html)
- [OVITO modifier reference](https://ovito.org/manual/reference/pipelines/modifiers/index.html) · [OVITO reference index](https://ovito.org/manual/reference/index.html)
- [VMD License](https://tcbg.illinois.edu/Research/vmd/current/LICENSE.html)
- [Relicensing MDAnalysis](https://mdanalysis.org/2022/11/07/relicensing)
- [tomviz](https://tomviz.org) · [tomviz GitHub](https://github.com/OpenChemistry/tomviz)
- [freud](https://github.com/glotzerlab/freud) · [pygfx](https://github.com/pygfx/pygfx) · [MolecularNodes](https://github.com/BradyAJohnston/MolecularNodes)
- [JLmol](https://jlmol.com) · [Atomsk links](https://atomsk.univ-lille.fr/links.php)
- [VIAMD](https://github.com/scanberg/viamd) · [mdlib](https://github.com/scanberg/mdlib)
