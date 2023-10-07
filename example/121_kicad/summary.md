https://docs.kicad.org/7.0/en/getting_started_in_kicad/getting_started_in_kicad.html

- **Intro to KiCad Version 7**
  - Open-source software for creating electronic circuit schematics and PCBs.
  - Supports integrated and standalone design workflows.
  - Offers utilities like PCB calculator, Gerber viewer, 3D viewer, and SPICE simulator.
  - Runs on multiple OS and supports up to 32 copper layers.

- **Downloading and Installing**
  - Available for Windows, macOS, and Linux.
  - Periodic stable releases and nightly builds for new features.
  - Nightly builds may have bugs.

- **Support**
  - Official user forum, Discord, and IRC channels for community support.

- **Basic Concepts and Workflow**
  - Schematic drawing followed by board layout.
  - Schematic contains symbols representing electronic components and their connections.
  - Board layout involves placement of component footprints and copper tracks.

- **KiCad Editors**
  - Separate windows for schematic, board layout, and editing symbols and footprints.
  - Large library of pre-made symbols and footprints available.

- **Project-based Workflow**
  - All design files and settings contained in a project folder.
  - Importance of keeping all project-related files together.

- **PCB Design Workflow**
  - Schematic is drawn first, followed by layout.
  - Schematic and board layout are cross-verified for consistency.
  - Final outputs generated for fabrication after passing Design Rules Check (DRC).

- **Tutorial Part 1: Project Setup**
  - Creating a new project from the Project Manager.

- **Tutorial Part 2: Schematic**
  - Initial setup of symbol library table.
  - Basics of schematic editor including panning, zooming, and toolbars.
  - Sheet setup, adding symbols, and selecting and moving objects.


- Object Selection: Use Shift+click to add, Ctrl+Shift+click to remove. Ctrl+click toggles selection. Drag selection also supported.
- Move/Rotate: 'M' to move, 'R' to rotate. 'G' also moves but retains wire connections.
- Deletion: Use the 'Del' key.
  
- Wiring:
  - Symbol pins have circles when not connected.
  - Use 'W' hotkey or button to add a wire.
  - Hovering over a pin auto-starts wire drawing.
  
- Add Power and Ground:
  - Use 'P' hotkey or button.
  - Connect with wires.
  
- Labels:
  - Good practice to label nets.
  - Same-named labels and power symbols are electrically connected.

- Annotation:
  - Auto-annotation by default.
  - Manual annotation possible.
  
- Symbol Properties:
  - Right-click to edit properties like LED color, battery voltage, etc.
  
- Footprint Assignment:
  - Choose appropriate footprints for components.
  - Multiple filtering options available.
  
- Electrical Rules Check (ERC):
  - Identifies common connection issues.
  - Errors can be ignored or addressed.
  
- Bill of Materials:
  - Optional, generated via Python scripts.
  
- PCB Editor:
  - Similar navigation to the Schematic editor.
  - Various tools for designing the PCB.
  
- Board Setup:
  - Set page size, title block.
  - Define stackup and design rules for manufacturing.


- PCB Design Layers: Simple designs start with two layers; complex projects may need more.
- Physical Stackup & Design Rules: Defaults usually suffice for guides but must be customized per PCB fab house capabilities for real projects.
- Net Classes: Sets of design rules for specific groups of nets. Using net classes automates the management of design rules.
- Update PCB from Schematic: Manually import components from schematic to layout. Must be done each time schematic is updated.
- Drawing Board Outline: Define the board by drawing an outline on the Edge.Cuts layer.
- Placing Footprints: Position components considering electrical requirements, avoiding Courtyard overlap, and simplifying routing.
- Routing Tracks: Connect component pads with copper traces, keeping in mind the front and back layers.
- Placing Copper Zones: Used mainly for ground and power connections. Not filled automatically; must be manually filled.
- Design Rule Checking (DRC): Validates the layout for errors like insufficient clearance or unconnected traces. Strongly advised before generating outputs.
- 3D Viewer: KiCad offers a 3D view for inspecting the completed PCB layout.



- Article covers using KiCad for PCB design, detailing various steps and features.
  - Raytracing mode available for accurate 3D rendering.
  - Most footprints come with 3D models, users can add their own.
- Final step: generate fabrication outputs.
  - Usually uses Gerber format for PCB fabricators.
  - Necessary layers like copper, board outline, soldermask, and silkscreen must be checked.
  - Generate drill files for hole locations.
- Custom symbols and footprints:
  - Adding a switch needs new symbol and footprint library.
  - Symbols and footprints organized into global and project-specific libraries.
  - Libraries can be managed under Preferences.
  - Library paths can use substitution variables for flexibility.
- Creating new symbol for SPST switch.
  - Pins added and edited. 
  - Graphical features like lines and circles used for presentation.
- Creating new footprints:
  - Properly sized and positioned pads added.
  - Annular ring sizes adjusted for easier soldering.
- Footprint graphics:
  - Includes part outline on fabrication layer, larger outline on silkscreen, and a courtyard.
  
  
  
- Y origin for fab outline calculated as -1.8 mm.
- Silkscreen outline adjusted by 0.11 mm from fab outline; new Y origin -1.91 mm.
- Courtyard outline drawn with 0.25 mm clearance.
- KiCad Library Conventions (KLC) guide mentioned for maintaining quality.
- Switch symbol modified to use new footprint by default.
- Schematic updated to include new switch; ERC run to check electrical rules.
- Board layout updated; unnecessary traces deleted, new traces routed, DRC run.
- Symbols can have preselected footprints and footprint filters.
- 3D models for footprints stored separately; STEP and VRML supported.
- FreeCAD and StepUp Workbench recommended for creating 3D models.
- Additional KiCad learning resources and contribution options mentioned.
