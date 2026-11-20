"""Julia sampling functions (sampling_toNexus.jl), included once per process."""
from pathlib import Path

from juliacall import Main as jl

jl.include(str(Path(__file__).resolve().parent.parent / "sampling_toNexus.jl"))
