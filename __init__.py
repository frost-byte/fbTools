"""Top-level package for fbTools."""

WEB_DIRECTORY = "./js"
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """frost-byte"""
__email__ = "brian@frost-byte.net"
__version__ = "1.9.1"
# Basic colors
R = RED     = "\033[31m"
G = GREEN   = "\033[32m"
Y = YELLOW  = "\033[33m"
B = BLUE    = "\033[34m"
M = MAGENTA = "\033[35m"
C = CYAN    = "\033[36m"
WH = WHITE   = "\033[37m"
RES = RESET   = "\033[0m"

# Bright variants
BC = BRIGHT_CYAN = "\033[96m"
BG = BRIGHT_GREEN = "\033[92m"
          
LOGO = rf"""
{BG}
   ___  __    {B}   ______                ___            {BG}
 ╱'___╲╱╲ ╲   {B}  ╱╲__  _╲              ╱╲_ ╲              {BG}
╱╲ ╲__╱╲ ╲ ╲____{B}╲╱_╱╲ ╲╱   ___     ___╲╱╱╲ ╲     ____  {BG}
╲ ╲ ,__╲╲ ╲ '__`╲{B}  ╲ ╲ ╲  ╱ __`╲  ╱ __`╲╲ ╲ ╲   ╱',__╲ {BG}
 ╲ ╲ ╲_╱ ╲ ╲ ╲L╲ ╲{B}  ╲ ╲ ╲╱╲ ╲L╲ ╲╱╲ ╲L╲ ╲╲_╲ ╲_╱╲__, `╲{BG}
  ╲ ╲_╲   ╲ ╲_,__╱{B}   ╲ ╲_╲ ╲____╱╲ ╲____╱╱╲____╲╱╲____╱{BG}
   ╲╱_╱    ╲╱___╱{B}     ╲╱_╱╲╱___╱  ╲╱___╱ ╲╱____╱╲╱___╱ {BG}

                🧊 frost-byte{B} Tools loaded!{RES}

                                          
"""

from .extension import FBToolsExtension
async def comfy_entrypoint() -> FBToolsExtension:
    return FBToolsExtension()

print(LOGO)
print(rf"""""")
