"""What the PyInstaller specs (packaging/windows/ewald.spec, packaging/macos/ewald.spec) leave
out of the standalone apps to keep them small.

VTK_UNUSED: VTK modules that neither ewald nor pyvista import (the full test suite imports 36 of
VTK's ~150 modules; see docs/developer/windows-release.md, "Size"): VTK-m/Viskores accelerators
(about 90 MB of libraries), readers for CFD, geoscience and engineering formats, VR/WebXR and
testing. pyvista loads its readers by name only when asked, so they cost nothing else; a notebook
that needs one of them runs in the online install or a pip install instead.
"""

VTK_UNUSED = [f"vtkmodules.{m}" for m in (
    "vtkAcceleratorsVTKmCore", "vtkAcceleratorsVTKmDataModel", "vtkAcceleratorsVTKmFilters",
    "vtkGeovisCore", "vtkIOAMR", "vtkIOAvmesh", "vtkIOCGNSReader", "vtkIOCONVERGECFD", "vtkIOCesium3DTiles",
    "vtkIOCityGML", "vtkIOERF", "vtkIOEngys", "vtkIOExodus", "vtkIOFDS", "vtkIOFLUENTCFF", "vtkIOGeoJSON",
    "vtkIOH5Rage", "vtkIOH5part", "vtkIOIOSS", "vtkIOLANLX3D", "vtkIOLSDyna", "vtkIOMINC", "vtkIOMotionFX",
    "vtkIOOMF", "vtkIOOggTheora", "vtkIOPIO", "vtkIOParallelExodus", "vtkIOParallelLSDyna", "vtkIOSQL",
    "vtkIOSegY", "vtkIOTRUCHAS", "vtkIOVPIC", "vtkIOVeraOut", "vtkIOVideo", "vtkIOXdmf2",
    "vtkRenderingVR", "vtkRenderingVRModels", "vtkRenderingWebXR", "vtkRenderingVolumeAMR",
    "vtkTestingRendering", "vtkTestingSerialization", "vtkWebCore", "vtkWebGLExporter",
)]
