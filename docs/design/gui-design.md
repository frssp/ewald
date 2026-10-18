# ewald GUI 디자인 (step 2 handoff)

Mockup: https://claude.ai/artifact/7s8U7t8Dea9PCxMxmXu7Zo (Analyze, Render 두 artboard, Play로 클릭 가능)

## 설계 원칙

OVITO 사용자가 바로 적응할 수 있도록 큰 골격(viewport 중앙, 오른쪽 command panel에 pipeline 위·properties 아래, 하단 timeline, data inspector)은 OVITO 3 convention을 그대로 따랐습니다. 그 위에 VESTA에서 결정 구조 작업에 필요한 조작(view along a/b/c/a*/b*/c*, 15° step rotation, style preset, boundary 범위, bond search pair table, 거리/각도 측정, elements.ini 색 import)을 얹었고, OVITO Basic에 없는 Pro 기능(Python script modifier, code generator, path-tracing renderer)을 무료로 GUI에 노출하는 것을 차별점으로 삼았습니다. 모든 GUI 동작은 같은 `ewald` Python API로 표현되며, console과 Render 화면의 "Python for this figure"가 그 계약을 사용자에게 보여 줍니다.

## Analyze workspace (Main)

1. Menu bar → main toolbar: Open/Save, Undo/Redo, workspace switch (Analyze | Render), command palette (Ctrl K, modifier·command·property·view 통합 검색), Compute device pill (CUDA/CPU), Render 버튼.
2. 왼쪽 tool rail (46 px): Pick, Orbit, Pan, Zoom, Measure distance, Measure angle. Measure는 viewport에 dashed line + 값 label(Å)로 표시.
3. Viewport toolbar: View along a b c a* b* c*, rotate ±15°, tilt, Ortho/Persp, Style preset(Space-filling / Ball & stick / Points), layout 1 or 4 (Top/Front/Left/Perspective, active viewport는 노란 outline), Zoom to fit.
4. Viewport: drag로 orbit, atom click으로 pick(노란 ring, status bar에 id·위치·coordination). Overlay layer로 frame/time text, color legend(현재 coloring mode에 따라 자동: CNA, Color coding bar, type, Selected), 좌하단 crystal axes tripod. Background Dark/White.
5. Timeline: first/prev/play/next/last, frame slider, frame/time, fps·stride·loop.
6. Bottom dock tabs: Python console(IPython, `pipe`/`data`/`vp` 미리 bind, console에서 append한 modifier가 pipeline panel에 즉시 반영), Data inspector(Particles/Bonds/Attributes/Tables/Cell, filter expression), Plots(table을 바로 plot: g(r), histogram), Log.
7. Command panel: pipeline selector + "Add modification" (카테고리: Analysis, Coloring, Modification, Selection, Crystal & volume, Python; filter 입력). Pipeline list는 OVITO처럼 Visual elements / Modifications(위가 마지막) / Data source 순서, 각 행에 eye toggle, 2글자 badge, 이름, status(실행 시간·선택 수·삭제 수).
8. Properties: 선택한 item별 editor. Python script modifier는 inline editor + Run(Ctrl Enter) + 결과 상태(성공/에러 위치) + "Parameters exposed to the GUI"(함수 인자 → 자동 위젯).

## Render workspace

Renderer card(OpenGL / OSPRay / OptiX·ANARI), samples·AO·shadow·DOF·outline, journal size preset(1-column 3.5 in, 2-column 7.2 in, slide, square @ 600 dpi), background(transparent 체커보드 / white / black), output file(이미지 또는 animation range·fps). 중앙 preview는 최종 aspect의 frame guide이며, overlay(Text label with [Time]/[Frame] 치환, color legend, scale bar, crystal axes)는 pt 단위로 지정. 오른쪽 "Python for this figure"가 설정에 따라 실시간 갱신되는 code generator.

## 구현 매핑 (PySide6 + PyVista)

| 영역 | Qt 구현 |
| --- | --- |
| 전체 shell | `QMainWindow`, workspace는 `QStackedWidget` 또는 dock layout 저장/복원(`saveState`) |
| Viewport | `pyvistaqt.QtInteractor`; quad는 `QGridLayout`에 interactor 4개, camera preset은 `view_vector` |
| Overlay | VTK `vtkTextActor`/`scalar_bar`(viewport), Render용은 Qt painter로 합성해 PDF vector 출력 가능 |
| Pipeline list | `QTreeView` + custom model(section 3개), eye는 checkable column |
| Properties | 각 Modifier class가 `param` spec을 노출 → generic form builder (Python script의 함수 인자도 동일 경로) |
| Console | `qtconsole.RichJupyterWidget` + in-process kernel, namespace에 `pipe`, `data`, `vp` |
| Data inspector | `QTableView` + numpy-backed model(lazy), filter는 ExpressionSelection 엔진 재사용 |
| Plots | `pyqtgraph`(빠름) 또는 matplotlib `FigureCanvasQTAgg` |
| Command palette | frameless `QDialog` + fuzzy match over action registry (menu action과 같은 `QAction` 목록) |
| Timeline | `QSlider` + `QTimer`; frame 계산은 worker thread(`QThreadPool`)로, UI는 마지막 요청만 반영 |

## 우선순위 제안

MVP: shell + viewport(1-up) + timeline + pipeline list/properties(기존 step-1 modifier) + console. 다음: view-along/step rotation, pick/measure, data inspector, Add modification palette. 그 다음: quad view, command palette, Render workspace(OpenGL 먼저, OSPRay/ANARI는 step 4).

Theme token: bg #14161A, panel #1B1E23, raised #23272E, border #30353D, text #E7E9EC, secondary #AEB4BE, muted #8C939E, accent #5AA9FF, selection outline #E8C547, pick #FFD84D. Font IBM Plex Sans/Mono.
