# Related work and references

The literature EWALD builds on or should learn from, grouped by topic, with what each work means
for EWALD's design. Use it when you design a feature (read the works of that area first), when
you write the JOSS paper (`paper/paper.bib` holds the entries the paper cites) and when you
review a port (cite the method paper in the docstring and in the manual page).

The bibliographic data were checked on 2026-10-24 against publisher, PubMed Central, arXiv or
author records, not written from memory. Where no DOI could be confirmed the entry says so; look
it up before citing it. Entries marked **(paper)** are in `paper/paper.bib`.

## Visualization and analysis programs

- **OVITO.** A. Stukowski, "Visualization and analysis of atomistic simulation data with
  OVITO, the Open Visualization Tool", *Modelling Simul. Mater. Sci. Eng.* 18, 015012 (2010),
  [doi:10.1088/0965-0393/18/1/015012](https://doi.org/10.1088/0965-0393/18/1/015012) **(paper)**.
  The model for the pipeline (source, modifiers, cached frames), the property and attribute
  names and the structure type colors (`pipeline.py`, `modifiers/`). Its source is GPLv3-or-MIT;
  EWALD's C++ ports are taken under MIT (`_csrc/README.md`).
- **DXA.** A. Stukowski, V. V. Bulatov, A. Arsenlis, "Automated identification and indexing of
  dislocations in crystal interfaces", *Modelling Simul. Mater. Sci. Eng.* 20, 085007 (2012),
  [doi:10.1088/0965-0393/20/8/085007](https://doi.org/10.1088/0965-0393/20/8/085007) **(paper)**.
  The dislocation extraction algorithm behind OVITO's DXA. EWALD does not reimplement it and
  calls the optional `ovito` package instead (`modifiers/ovito_bridge.py`).
- **VESTA 3.** K. Momma, F. Izumi, "VESTA 3 for three-dimensional visualization of crystal,
  volumetric and morphology data", *J. Appl. Cryst.* 44, 1272-1276 (2011),
  [doi:10.1107/S0021889811038970](https://doi.org/10.1107/S0021889811038970) **(paper)**.
  The reference for crystal display: boundary ghosts (`boundary.py`), polyhedra, isosurfaces of
  CHGCAR data, the element colors in `tables/colors.json`.
- **VMD.** W. Humphrey, A. Dalke, K. Schulten, "VMD: Visual molecular dynamics",
  *J. Mol. Graph.* 14, 33-38 (1996),
  [doi:10.1016/0263-7855(96)00018-5](https://doi.org/10.1016/0263-7855(96)00018-5) **(paper)**.
  Representations, selections and scripting for large trajectories. Its license rules out
  taking code; ideas only.
- **Jmol.** R. M. Hanson, "Jmol, a paradigm shift in crystallographic visualization",
  *J. Appl. Cryst.* 43, 1250-1260 (2010),
  [doi:10.1107/S0021889810030256](https://doi.org/10.1107/S0021889810030256).
  A scriptable crystal viewer used in teaching; the case for a viewer whose every state is a
  command, as EWALD's `pipe.view`.
- **Avogadro.** M. D. Hanwell, D. E. Curtis, D. C. Lonie, T. Vandermeersch, E. Zurek,
  G. R. Hutchison, "Avogadro: an advanced semantic chemical editor, visualization, and analysis
  platform", *J. Cheminform.* 4, 17 (2012),
  [doi:10.1186/1758-2946-4-17](https://doi.org/10.1186/1758-2946-4-17).
  Plugin architecture of a desktop molecular editor; a reference for structure editing when the
  owner asks for it (deferred).
- **XCrySDen.** A. Kokalj, *J. Mol. Graphics Modelling* 17, 176-179 (1999) (DOI not
  confirmed). Crystal structures, Fermi surfaces and densities from DFT codes; the classic
  companion of plane-wave codes.
- **NGL and nglview.** A. S. Rose, A. R. Bradley, Y. Valasatava, J. M. Duarte, A. Prlić,
  P. W. Rose, "NGL viewer: web-based molecular graphics for large complexes", *Bioinformatics*
  (2018), [doi:10.1093/bioinformatics/bty419](https://doi.org/10.1093/bioinformatics/bty419);
  H. Nguyen, D. A. Case, A. S. Rose, "NGLview, interactive molecular graphics for Jupyter
  notebooks", *Bioinformatics* 34, 1241-1242 (2018),
  [doi:10.1093/bioinformatics/btx789](https://doi.org/10.1093/bioinformatics/btx789) **(paper)**.
  Millions of atoms in a browser with WebGL; the Jupyter widget EWALD's `display.py` is compared
  with.
- **chemiscope.** G. Fraux, R. K. Cersonsky, M. Ceriotti, *J. Open Source Softw.* 5, 2117
  (2020), [doi:10.21105/joss.02117](https://doi.org/10.21105/joss.02117) **(paper)**.
  Structure-property maps linked to a 3D view: the idea behind `linking.py` (plots and the view
  share a selection).
- **VIAMD.** R. Skånberg, I. Hotz, A. Ynnerman, M. Linares, "VIAMD: a software for visual
  interactive analysis of molecular dynamics", *J. Chem. Inf. Model.* 63, 7382-7391 (2023),
  [doi:10.1021/acs.jcim.3c01033](https://doi.org/10.1021/acs.jcim.3c01033).
  Interactive MD analysis with a scripting interface for user-defined properties; a fine reference
  (CLAUDE.md, "Reuse first") for linked selection and per-frame properties.
- **MegaMol.** S. Grottel, M. Krone, C. Müller, G. Reina, T. Ertl, "MegaMol, a prototyping
  framework for particle-based visualization", *IEEE TVCG* 21, 201-214 (2015),
  [doi:10.1109/TVCG.2014.2350479](https://doi.org/10.1109/TVCG.2014.2350479).
  Research framework for very large particle data, from the group behind the culling work in
  the rendering section.
- **tomviz.** J. Schwartz et al., "Real-time 3D analysis during electron tomography using
  tomviz", *Nat. Commun.* 13, 4458 (2022),
  [doi:10.1038/s41467-022-32046-0](https://doi.org/10.1038/s41467-022-32046-0).
  Python operators in a VTK desktop application that update while data still arrive; the same
  shape as EWALD's growing remote trajectories and Python steps.
- **VTK and PyVista.** W. Schroeder, K. Martin, B. Lorensen, *The Visualization Toolkit*, 4th ed.,
  Kitware (2006) **(paper)**; C. B. Sullivan, A. A. Kaszynski, *J. Open Source Softw.* 4, 1450
  (2019), [doi:10.21105/joss.01450](https://doi.org/10.21105/joss.01450) **(paper)**. The
  interactive viewport and offscreen OpenGL renderer (`render.py`, `gui/viewport.py`).

## Analysis libraries and methods

- **ASE.** A. H. Larsen et al., *J. Phys.: Condens. Matter* 29, 273002 (2017),
  [doi:10.1088/1361-648X/aa680e](https://doi.org/10.1088/1361-648X/aa680e) **(paper)**.
  Fallback reader and writer for every format without a native reader (`io/ase_reader.py`),
  and the "Simple (ASE)" display style.
- **pymatgen.** S. P. Ong et al., *Comput. Mater. Sci.* 68, 314-319 (2013),
  [doi:10.1016/j.commatsci.2012.10.028](https://doi.org/10.1016/j.commatsci.2012.10.028)
  **(paper)**. Source of the radius tables (`tables/make_radii.py`) and of oxidation-state
  guessing for Shannon radii.
- **MDAnalysis.** N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, O. Beckstein,
  *J. Comput. Chem.* 32, 2319-2327 (2011),
  [doi:10.1002/jcc.21787](https://doi.org/10.1002/jcc.21787) **(paper)**. XTC/TRR/DCD readers
  (`io/mdanalysis_reader.py`) and a reader benchmark baseline.
- **MDTraj.** R. T. McGibbon et al., "MDTraj: a modern open library for the analysis of
  molecular dynamics trajectories", *Biophys. J.* 109, 1528-1532 (2015),
  [doi:10.1016/j.bpj.2015.08.015](https://doi.org/10.1016/j.bpj.2015.08.015) **(paper)**.
  The MDTraj HDF5 format (`io/mdtraj_h5.py`) and a design point for fast trajectory I/O.
- **freud.** V. Ramasubramani et al., *Comput. Phys. Commun.* 254, 107275 (2020),
  [doi:10.1016/j.cpc.2020.107275](https://doi.org/10.1016/j.cpc.2020.107275) **(paper)**.
  Independent reference values for Steinhardt order, RDF and neighbor analyses
  (`tests/data/analysis_reference.npz`).
- **spglib.** A. Togo, K. Shinohara, I. Tanaka, *Sci. Technol. Adv. Mater. Methods* 4, 2384822
  (2024), [doi:10.1080/27660400.2024.2384822](https://doi.org/10.1080/27660400.2024.2384822)
  **(paper)**. Symmetry analysis.
- **phonopy.** A. Togo, L. Chaput, T. Tadano, I. Tanaka, "Implementation strategies in phonopy
  and phono3py", *J. Phys.: Condens. Matter* 35, 353001 (2023),
  [doi:10.1088/1361-648X/acd831](https://doi.org/10.1088/1361-648X/acd831) **(paper)**.
  Phonon modes and bands, in the `ewald-phonopy` plugin.
- **PLUMED 2.** G. A. Tribello, M. Bonomi, D. Branduardi, C. Camilloni, G. Bussi, "PLUMED 2:
  New feathers for an old bird", *Comput. Phys. Commun.* 185, 604-613 (2014),
  [doi:10.1016/j.cpc.2013.09.018](https://doi.org/10.1016/j.cpc.2013.09.018) **(paper)**.
  Collective variables, in the `ewald-plumed` plugin.
- **AiiDA 1.0.** S. P. Huber et al., "AiiDA 1.0, a scalable computational infrastructure for
  automated reproducible workflows and data provenance", *Sci. Data* 7, 300 (2020),
  [doi:10.1038/s41597-020-00638-4](https://doi.org/10.1038/s41597-020-00638-4) **(paper)**.
  Provenance graphs and workchains, in the `ewald-aiida` plugin; the provenance model is also
  a reference for what an `.ewd` bundle should record.
- **H5MD.** P. de Buyl, P. H. Colberg, F. Höfling, "H5MD: a structured, efficient, and
  portable file format for molecular data", *Comput. Phys. Commun.* 185, 1546-1553 (2014),
  [doi:10.1016/j.cpc.2014.01.018](https://doi.org/10.1016/j.cpc.2014.01.018). The H5MD
  reader and writer (`io/h5md.py`).
- **Common neighbor analysis.** J. D. Honeycutt, H. C. Andersen, *J. Phys. Chem.* 91,
  4950-4963 (1987), [doi:10.1021/j100303a014](https://doi.org/10.1021/j100303a014)
  **(paper)**; the adaptive and interval variants: A. Stukowski, "Structure identification
  methods for atomistic simulations of crystalline materials", *Modelling Simul. Mater. Sci.
  Eng.* 20, 045021 (2012),
  [doi:10.1088/0965-0393/20/4/045021](https://doi.org/10.1088/0965-0393/20/4/045021)
  **(paper)**. The C++ CNA port; the second paper also compares CNA, CSP, bond-angle and
  Voronoi methods, a good guide for which default to offer.
- **Polyhedral template matching.** P. M. Larsen, S. Schmidt, J. Schiøtz, *Modelling Simul.
  Mater. Sci. Eng.* 24, 055007 (2016),
  [doi:10.1088/0965-0393/24/5/055007](https://doi.org/10.1088/0965-0393/24/5/055007)
  **(paper)**. The vendored `ptm` library.
- **Centrosymmetry parameter.** C. L. Kelchner, S. J. Plimpton, J. C. Hamilton, "Dislocation
  nucleation and defect structure during surface indentation", *Phys. Rev. B* 58, 11085-11088
  (1998), [doi:10.1103/PhysRevB.58.11085](https://doi.org/10.1103/PhysRevB.58.11085)
  **(paper)**. The CSP modifier.
- **Bond-orientational order.** P. J. Steinhardt, D. R. Nelson, M. Ronchetti, *Phys. Rev. B*
  28, 784-805 (1983), [doi:10.1103/PhysRevB.28.784](https://doi.org/10.1103/PhysRevB.28.784)
  **(paper)**. Steinhardt Q_l/W_l.
- **Voronoi cells.** C. H. Rycroft, "VORO++: a three-dimensional Voronoi cell library in C++",
  *Chaos* 19, 041111 (2009), [doi:10.1063/1.3215722](https://doi.org/10.1063/1.3215722)
  **(paper)**. The vendored voro++.

## 3D rendering

EWALD has three renderers (CLAUDE.md, "Layout"): the VTK viewport with GPU sphere impostors
above 150k atoms and SSAO, the numba ray tracer (`raytrace.py`: uniform grid for atoms, binned
SAH BVH for large meshes, hemisphere ambient occlusion, shadow rays, Blinn-Phong, adaptive
sampling), and the matplotlib vector renderer. These are the works each part follows.

- **Survey.** B. Kozlíková, M. Krone, M. Falk, N. Lindow, M. Baaden, D. Baum, I. Viola,
  J. Parulek, H.-C. Hege, "Visualization of biomolecular structures: state of the art
  revisited", *Comput. Graph. Forum* 36(8), 178-204 (2017),
  [doi:10.1111/cgf.13072](https://doi.org/10.1111/cgf.13072).
  Start here: a taxonomy of molecular rendering (space-filling, ball-and-stick, surfaces),
  level of detail, and how dynamic data are shown. Most of its methods carry over to crystals.
- **Sphere and cylinder impostors.** C. Sigg, T. Weyrich, M. Botsch, M. Gross, "GPU-based
  ray-casting of quadratic surfaces", *Eurographics Symposium on Point-Based Graphics*, 59-65
  (2006), [doi:10.2312/SPBG/SPBG06/059-065](https://doi.org/10.2312/SPBG/SPBG06/059-065).
  Per-pixel ray casting of quadrics from one vertex: the basis of the impostor shader in
  `render.py`. Cylinders (bonds) can be drawn the same way, which would lift the mesh limit on
  bonds in large systems.
- **Ambient occlusion and edge cueing for molecules (QuteMol).** M. Tarini, P. Cignoni,
  C. Montani, "Ambient occlusion and edge cueing to enhance real time molecular visualization",
  *IEEE TVCG* 12(5), 1237-1244 (2006),
  [doi:10.1109/TVCG.2006.115](https://doi.org/10.1109/TVCG.2006.115) **(paper)**.
  Per-atom ambient occlusion computed once and stored, impostors for space-fill and
  ball-and-stick, and depth-aware outlines. For EWALD: an alternative to screen-space SSAO in the
  impostor path that costs nothing per frame for a static structure, and the look the ray
  tracer's AO should match in the viewport.
- **Feature lines from a G-buffer.** T. Saito, T. Takahashi, "Comprehensible rendering of 3-D
  shapes", *Proc. SIGGRAPH* 1990, *Comput. Graph.* 24, 197-206,
  [doi:10.1145/97880.97901](https://doi.org/10.1145/97880.97901).
  Silhouettes and creases from depth and normal buffers in image space. The ASE-style outlines
  in `render.py` are geometric (inverted hulls); a depth-buffer pass would give outlines for
  every style and renderer, including illustrative figures.
- **Culling and deferred shading for large MD.** S. Grottel, G. Reina, C. Dachsbacher, T. Ertl,
  "Coherent culling and shading for large molecular dynamics visualization",
  *Comput. Graph. Forum* 29(3), 953-962 (2010),
  [doi:10.1111/j.1467-8659.2009.01698.x](https://doi.org/10.1111/j.1467-8659.2009.01698.x).
  Two-level occlusion culling plus GPU ray casting for tens of millions of glyphs. The next
  step after LOD when `viewport.*` budgets are not enough for multi-million-atom frames.
- **Ray tracing for molecular graphics at scale.** J. E. Stone et al., "Atomic detail
  visualization of photosynthetic membranes with GPU-accelerated ray tracing", *Parallel
  Comput.* 55, 17-27 (2016),
  [doi:10.1016/j.parco.2015.10.015](https://doi.org/10.1016/j.parco.2015.10.015).
  GPU ray tracing in VMD for publication images and movies of a very large system; the quality
  bar for EWALD's Publication preset.
- **OSPRay.** I. Wald, G. P. Johnson, J. Amstutz, C. Brownlee, A. Knoll, J. Jeffers,
  J. Günther, P. Navrátil, "OSPRay, a CPU ray tracing framework for scientific visualization",
  *IEEE TVCG* 23(1), 931-940 (2017) (DOI not confirmed). CPU ray tracing as a production
  renderer for visualization; the reason a CPU ray tracer is a sound choice for EWALD (pip VTK
  has no OSPRay, so `raytrace.py` fills that role).
- **BVH construction.** I. Wald, "On fast construction of SAH-based bounding volume
  hierarchies", *IEEE Symposium on Interactive Ray Tracing* (2007) (DOI not confirmed).
  Binned SAH build, as in `raytrace.build_bvh`.
- **Adaptive and nonuniform sampling.** D. P. Mitchell, "Generating antialiased images at low
  sampling densities", *Proc. SIGGRAPH* 1987, *Comput. Graph.* 21(4), 65-72,
  [doi:10.1145/37402.37410](https://doi.org/10.1145/37402.37410).
  Extra samples only where the image needs them; the idea behind the ray tracer's early stop of
  converged pixels.
- **Textbook.** M. Pharr, W. Jakob, G. Humphreys, *Physically Based Rendering: From Theory to
  Implementation*, 4th ed., MIT Press (2023), ISBN 978-0-262-04802-6 (free online edition at
  pbr-book.org). Reference for BVHs, sampling, shading and filters when the ray tracer changes.

## Notebooks and reproducibility

- **Jupyter.** T. Kluyver et al., "Jupyter Notebooks, a publishing format for reproducible
  computational workflows", in *Positioning and Power in Academic Publishing*, IOS Press, 87-90
  (2016), [doi:10.3233/978-1-61499-649-1-87](https://doi.org/10.3233/978-1-61499-649-1-87)
  **(paper)**. The notebook model of the GUI, the kernel bridge and `.ipynb` export.
- **Ten simple rules for notebooks.** A. Rule et al., "Ten simple rules for writing and sharing
  computational analyses in Jupyter Notebooks", *PLoS Comput. Biol.* 15, e1007007 (2019),
  [doi:10.1371/journal.pcbi.1007007](https://doi.org/10.1371/journal.pcbi.1007007) **(paper)**.
  For example: tell a story for an audience, and use cell divisions to keep steps clear. EWALD's
  step cells (one `pipe.add` each) and the plain `.py` format follow much of this by
  construction; the rules are also a checklist for the notebooks the AI writes.
- **Exploration versus explanation.** A. Rule, A. Tabard, J. D. Hollan, "Exploration and
  explanation in computational notebooks", *Proc. CHI* 2018, paper 32,
  [doi:10.1145/3173574.3173606](https://doi.org/10.1145/3173574.3173606).
  One in four of over a million GitHub notebooks had no explanatory text. For EWALD: a notebook that
  the GUI and the AI write should carry its own explanation (markdown cells, the narrator).
- **Notebooks in practice.** M. B. Kery, M. Radensky, M. Arya, B. E. John, B. A. Myers, "The
  story in the notebook: exploratory data science using a literate programming tool",
  *Proc. CHI* 2018, article 174,
  [doi:10.1145/3173574.3173748](https://doi.org/10.1145/3173574.3173748).
  How analysts keep, copy and discard cells while exploring; relevant to undo, cell history and
  how the AI edits existing cells instead of appending new ones.
- **Reproducibility of notebooks.** J. F. Pimentel, L. Murta, V. Braganholo, J. Freire, "A
  large-scale study about quality and reproducibility of Jupyter notebooks", *Proc. MSR* 2019
  (DOI not confirmed; data at
  [doi:10.5281/zenodo.3519618](https://doi.org/10.5281/zenodo.3519618)). Reruns public
  notebooks at scale and reports why they fail; the case for EWALD's one-namespace,
  top-to-bottom run and the `.ewd` bundle with data and plugins.
- **FAIR.** M. D. Wilkinson et al., "The FAIR Guiding Principles for scientific data
  management and stewardship", *Sci. Data* 3, 160018 (2016),
  [doi:10.1038/sdata.2016.18](https://doi.org/10.1038/sdata.2016.18).
  What an exported `.ewd` should satisfy: identifiers and checksums for its data, open formats,
  machine-readable metadata.
- **pyiron.** J. Janssen et al., "pyiron: an integrated development environment for
  computational materials science", *Comput. Mater. Sci.* 163, 24-36 (2019),
  [doi:10.1016/j.commatsci.2018.07.043](https://doi.org/10.1016/j.commatsci.2018.07.043).
  The closest prior work on the notebook side: Jupyter as the IDE for materials simulation with
  job and data management. pyiron runs calculations; EWALD analyses and shows them, so the two
  could meet through a workflow source (`workflow.py`).

## AI agents and tools

EWALD's AI acts only through notebook cells (CLAUDE.md, "Security"; project memory "AI only via
notebook"). The works below are the ideas that design stands on and the ones still to use.

- **Tool-using chemistry agents.** A. M. Bran, S. Cox, O. Schilter, C. Baldassari, A. D. White,
  P. Schwaller, "Augmenting large language models with chemistry tools", *Nat. Mach. Intell.*
  (2024), [doi:10.1038/s42256-024-00832-8](https://doi.org/10.1038/s42256-024-00832-8)
  **(paper)** (ChemCrow: GPT-4 with 18 expert tools); D. A. Boiko, R. MacKnight, B. Kline,
  G. Gomes, "Autonomous chemical research with large language models", *Nature* 624, 570-578
  (2023), [doi:10.1038/s41586-023-06792-0](https://doi.org/10.1038/s41586-023-06792-0)
  **(paper)** (Coscientist: planning, documentation search, code execution and lab
  automation). The two references for "an LLM with domain tools does science tasks". Both lean
  on documentation search and small, well-described tools; EWALD's `python_api` tool and
  `llms.txt` recipes follow the same idea.
- **Agents for MD.** Q. Campbell, S. Cox, J. Medina, B. Watterson, A. D. White, "MDCrow:
  automating molecular dynamics workflows with large language models", arXiv:2502.09565 (2025),
  ICLR 2025 AgenticAI workshop **(paper)**. 40 tools for setting up, running and analysing MD,
  evaluated on 25 tasks of graded difficulty. The closest agent to EWALD's domain; its task
  set is a model for harder `ewald eval` tasks (the suite is saturated, docs/todo.md).
- **Agents for materials data.** Y. Chiang, E. Hsieh, C.-H. Chou, J. Riebesell, "LLaMP: large
  language model made powerful for high-fidelity materials knowledge retrieval and
  distillation", arXiv:2401.17244 (2024). Hierarchical ReAct agents over the Materials Project
  with grounding to reduce hallucination; an idea for an AI tool that fetches reference
  structures and properties.
- **LLMs in materials science, survey of uses.** K. M. Jablonka et al., "14 examples of how LLMs
  can transform materials science and chemistry: a reflection on a large language model
  hackathon", *Digital Discovery* 2, 1233-1250 (2023),
  [doi:10.1039/D3DD00113J](https://doi.org/10.1039/D3DD00113J).
  A catalogue of ideas: property prediction, tool interfaces, extraction from papers,
  teaching assistants; several map to EWALD plugins (paper-to-modifier, the narrator).
- **Reason and act.** S. Yao et al., "ReAct: synergizing reasoning and acting in language
  models", arXiv:2210.03629 (2022), ICLR 2023. Interleaved reasoning and tool calls; the loop in
  `llm/agent.py`.
- **Self-taught tool use.** T. Schick et al., "Toolformer: language models can teach themselves
  to use tools", arXiv:2302.04761 (2023). When and how a model decides to call a tool.
- **Code as the action space.** X. Wang et al., "Executable code actions elicit better LLM
  agents", arXiv:2402.01030 (2024) **(paper)** (CodeAct). Agents that act by writing Python
  instead of JSON tool calls do better and can revise earlier actions from the results. This is the strongest
  outside support for EWALD's rule that the AI changes results only through notebook cells: a
  cell is a code action the user can read and rerun.
- **Agent-computer interfaces.** J. Yang et al., "SWE-agent: agent-computer interfaces enable
  automated software engineering", arXiv:2405.15793 (2024). Small, purpose-built commands with
  concise feedback help agents more than a raw shell. For EWALD's tools: short outputs, explicit errors, guards
  against edits that break a cell.
- **Visualization by an LLM.** T. Mallick, O. Yildiz, D. Lenz, T. Peterka, "ChatVis: automating
  scientific visualization with a large language model", arXiv:2410.11863 (2024). An LLM
  writes ParaView Python scripts, and execution errors are fed back until the script runs. The
  same loop as EWALD's Modifier Forge (`llm/forge.py`); its benchmark of visualization tasks is
  a source for eval tasks about the 3D view.
- **Natural-language interfaces for visualization.** L. Shen et al., "Towards natural language
  interfaces for data visualization: a survey", arXiv:2109.03506 (2021), published in
  *IEEE TVCG* (DOI not confirmed). Query interpretation, visual mapping, dialogue management:
  a checklist for what Ask AI should handle about views and plots.
- **AI assistants inside notebooks.** A. M. McNutt, C. Wang, R. A. DeLine, S. M. Drucker, "On the
  design of AI-powered code assistants for notebooks", *Proc. CHI* 2023,
  [doi:10.1145/3544548.3580940](https://doi.org/10.1145/3544548.3580940).
  Interviews with 15 data scientists: ask to disambiguate visualization requests, keep tools
  narrow, value polite assistants. Directly relevant to the chat side bar and to how the
  AI inserts and edits cells.
- **Human-AI interaction guidelines.** S. Amershi et al., "Guidelines for human-AI
  interaction", *Proc. CHI* 2019,
  [doi:10.1145/3290605.3300233](https://doi.org/10.1145/3290605.3300233).
  18 guidelines (make clear what the system can do, support efficient correction, show why it
  acted). Following cells as the AI writes them (D35) and undo of AI changes are instances.
- **Mixed-initiative interfaces.** E. Horvitz, "Principles of mixed-initiative user
  interfaces", *Proc. CHI* 1999, 159-166 (DOI not confirmed). When an assistant should act on
  its own and when it should ask; the background for proactive suggestions.
- **Prompt injection.** K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, M. Fritz,
  "Not what you've signed up for: compromising real-world LLM-integrated applications with
  indirect prompt injection", arXiv:2302.12173 (2023) (published at AISec 2023; DOI not
  confirmed). Instructions hidden in retrieved data take over an agent. The threat model behind
  `ewald/trust.py`: comments in structure files, notebooks and bundles are data until trusted.

## Figures and color

- **Better figures.** N. P. Rougier, M. Droettboom, P. E. Bourne, "Ten simple rules for better
  figures", *PLoS Comput. Biol.* 10, e1003833 (2014),
  [doi:10.1371/journal.pcbi.1003833](https://doi.org/10.1371/journal.pcbi.1003833).
  "Do not trust the defaults": the reason for the publication plot style (`plotstyle.py`) and
  the user `.mplstyle` (D39).
- **Color maps.** F. Crameri, G. E. Shephard, P. J. Heron, "The misuse of colour in science
  communication", *Nat. Commun.* 11, 5444 (2020),
  [doi:10.1038/s41467-020-19160-7](https://doi.org/10.1038/s41467-020-19160-7) **(paper)**;
  J. R. Nuñez, C. R. Anderton, R. S. Renslow, "Optimizing colormaps with consideration for color
  vision deficiency to enable accurate interpretation of scientific data", *PLoS ONE* 13,
  e0199239 (2018), [doi:10.1371/journal.pone.0199239](https://doi.org/10.1371/journal.pone.0199239)
  (cividis). Perceptually uniform, color-vision-deficiency-safe maps as defaults; no rainbow
  (jet) by default in `colormaps.py` and color bars.
- **Matplotlib.** J. D. Hunter, "Matplotlib: a 2D graphics environment", *Comput. Sci. Eng.* 9,
  90-95 (2007), [doi:10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55) **(paper)**.
  Plots and the vector renderer (`vector.py`).

## Ideas from this list

Open items that these works suggest are in docs/todo.md under "From the literature". In short:
precomputed per-atom ambient occlusion and image-space outlines in the viewport (Tarini et al.;
Saito and Takahashi), cylinder impostors for bonds (Sigg et al.), occlusion culling for
multi-million-atom frames (Grottel et al.), harder AI eval tasks modelled on MDCrow and ChatVis,
and an AI tool that fetches reference structures (LLaMP).
