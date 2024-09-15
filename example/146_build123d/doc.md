## build123d Documentation Summary

- This is a summary of the documentation from 
https://build123d.readthedocs.io/en/latest/
- Generated 2024-09-15

This documentation describes build123d, a Python-based parametric CAD system. Here is a summary of the main topics covered:

**1. Introduction**
    - Compares build123d to other CAD systems like SolidWorks, OnShape, Fusion 360, Blender, and OpenSCAD.
    - Highlights build123d's advantages: Boundary Representation (BREP) modeling, parameterized models, use of the Python programming language, open-source nature, and improvements over CadQuery.
    - Emphasizes the benefits of source code control systems, automated testing, and documentation. 

**2. Installation**
    - Explains installation using pip or from the GitHub repository.
    - Offers a development install guide.
    - Includes special notes for Apple Silicon installations.

**3. Key Concepts (Builder and Algebra Modes)**
    - **Builder mode:** 
        - Explains basic topological objects: Vertex, Edge, Wire, Face, Shell, Solid, Compound, and Shape.
        - Introduces the concept of "Builders" (BuildLine, BuildSketch, BuildPart) and their functionalities.
        - Describes the use of workplanes and "Location" contexts to position objects.
        - Discusses the use of combination modes (`Mode.ADD`, `Mode.SUBTRACT`, etc.)
        - Explores object selection using Selectors, along with practical examples. 
    - **Algebra mode:** 
        - Introduces the concept of object arithmetic for fusing, cutting, and intersecting parts.
        - Explains placement arithmetic using the `*` operator to locate objects relative to planes and existing locations.
        - Provides examples combining object arithmetic with placement arithmetic.

**4. Objects**
    - Explains `Align` parameter to control object placement relative to axes.
    - Introduces the `Mode` parameter in builder mode to define object combination behaviors.
    - Showcases available 1D, 2D, and 3D objects with illustrations and descriptions. 
    - Offers examples of custom object creation by subclassing the base object classes.

**5. Operations**
    - Defines operations as functions acting upon CAD objects to create new ones (e.g. `extrude`).
    - Lists available operations categorized by applicability (0D, 1D, 2D, 3D).
    - Offers examples demonstrating various operations in Builder and Algebra mode. 

**6. Builders**
    - **BuildLine**: Creates one-dimensional objects for sketches and paths. Explains basic functionalities, working with other builders and working on different planes. 
    - **BuildSketch**: Creates two-dimensional objects primarily used as profiles for 3D operations.  Highlights working on different planes and positioning objects. 
    - **BuildPart**: Creates three-dimensional objects and discusses handling implicit parameters and units.

**7. Joints**
    - Describes the use of `Joint` objects to assemble and position solid objects. 
    - Explains the functionality and proper pairing of various joints: RigidJoint, RevoluteJoint, LinearJoint, CylindricalJoint, and BallJoint.
    - Offers a detailed step-by-step tutorial on creating and connecting joints.

**8. Assemblies**
    - Explains the concept of assemblies for combining multiple parts into a `Compound` object.
    - Describes the importance of labeling individual components within an assembly. 
    - Discusses the difference between shallow and deep copies of shapes and their impact on performance. 

**9. Tutorials**
    - **Selectors Tutorial:** Guides the user through various methods for identifying and selecting CAD features using Build123d.
    - **Lego Tutorial:** Provides a comprehensive step-by-step guide to creating a parametric LEGO block model.
    - **Joints Tutorial:**  Illustrates the use of joints by creating a box with a hinged lid using RigidJoint, RevoluteJoint, and CylindricalJoint types.
    - **Surface Modeling Tutorial:**  Covers creating non-planar surfaces and objects by building a game token using Face.make_surface.
    - **Too Tall Toby (TTT) Tutorials:**  Presents a series of CAD challenges provided by Too Tall Toby to enhance Build123d proficiency.

**10. Import/Export**
    - Lists the supported file formats for import/export (3MF, BREP, DXF, glTF, STL, STEP, and SVG) with brief descriptions of each. 
    - Describes the use of the 2D exporters (`ExportDXF`, `ExportSVG`) for creating drawings from 3D objects by using projections and sections. 
    - Introduces the `LineType` enum for customizing line appearances in drawings. 
    - Explains 3D mesh export and import functionalities with the `Mesher` class.

**11. Tips, Best Practices, and FAQ**
    - Offers tips and tricks for successful 3D modeling in build123d.
    - Discusses frequently encountered issues and their solutions.
    - Covers important aspects like the choice of import statements and 
     the rationale behind sketching always occurring on a local `Plane.XY`.
    - Provides insights into handling workplane inconsistencies within nested Builders. 

**12. Cheat Sheet**
    - Summarizes all important concepts in build123d: 
        - Lists available Builders, Object types, Operations, Selectors, and Selector Operators.
        - Includes edge, wire, shape, plane, vector, and vertex operator descriptions.
        - Lists all available Enums for defining parameters and options in build123d. 

**13. Advanced Topics**
    - **Algebra performance**: Explores performance considerations in Algebra mode, advocating for lazy evaluation and vectorized operations.
    - **Location arithmetic**: Delves into precise positioning of shapes using Locations and Planes in Algebra mode. 
    - **Algebra definition**: Presents a mathematical framework of Build123d Algebra operations and notations. 
    - **Center**:  Details various methods (bounding box, geometry, mass) to find the center of CAD objects using the `CenterOf` enum.
    - **Debugging & logging**: Discusses techniques for debugging and logging within a build123d script using Python's debugger and logging facilities.

**14. External Tools and Libraries**
    - Presents a list of external tools and libraries useful for building, viewing, importing/exporting and managing Build123d designs. 
    - Introduces popular viewers like `ocp-vscode`, `cq-editor` fork and `yet-another-cad-viewer`.
    - Discusses part libraries such as `bd_warehouse`, and advanced tools like `blendquery`, `nething`, `ocp-freecad-cam` and `PartCAD`.

**15.  Examples**
    - Provides several example build123d scripts demonstrating both builder and algebra modes for building a variety of objects.  
    - Examples showcase many of the features of build123d (lofting, sweeping, importing and working on different planes) and are a valuable
      resource for learning how to design complex objects.  


**Overall, this documentation covers Build123d in depth and caters to both novice and advanced users with numerous practical examples, tutorials and tips.**
