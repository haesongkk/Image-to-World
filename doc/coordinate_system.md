# Coordinate System

## Purpose
This document explains the pseudo-world coordinate conventions used by the `compose_layout` and `assemble_scene` stages.

## Image To Pseudo World
- `compose_layout` interprets image center as camera principal center.
- Positive `X` moves right in the image plane.
- Positive `Y` moves upward in pseudo-world space, so image Y is inverted.
- Positive `Z` is farther from the camera inside `compose_layout`.

## Scene Assembly Export
- `assemble_scene` flips the sign of `Z` before exporting meshes.
- This keeps the exported OBJ more convenient for the current viewer/tooling assumptions.

## Scale
- Object scale is heuristic and derived from 2D bounding-box extent plus relative depth.
- It is not metric and should be treated as initialization only.

## Follow-up
Recommended future work:
- recover camera intrinsics/extrinsics
- estimate floor plane
- normalize asset scale against semantic object priors
