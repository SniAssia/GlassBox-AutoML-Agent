"""Build your GlassBox library to WASM"""
import pyodide_build

# This will package your library for WASM
pyodide_build.buildpkg.create_package(".")