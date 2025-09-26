from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field


class Cut(str, Enum):
    Fair = "Fair"
    Good = "Good"
    VeryGood = "Very Good"
    Premium = "Premium"
    Ideal = "Ideal"


class Color(str, Enum):
    J = "J"
    I = "I"
    H = "H"
    G = "G"
    F = "F"
    E = "E"
    D = "D"


class Clarity(str, Enum):
    I1 = "I1"
    SI2 = "SI2"
    SI1 = "SI1"
    VS2 = "VS2"
    VS1 = "VS1"
    VVS2 = "VVS2"
    VVS1 = "VVS1"
    IF = "IF"


# class DiamondRaw(BaseModel):
#     # Price is 326
#     carat: Annotated[float, Field(strict=True, gt=0)] = 0.23
#     cut: Cut = Cut.Ideal
#     color: Color = Color.E
#     clarity: Clarity = Clarity.SI2
#     depth: Annotated[float, Field(strict=True, gt=0)] = 61.5
#     table: Annotated[float, Field(strict=True, gt=0)] = 55
#     x: Annotated[float, Field(strict=True, gt=0)] = 3.95
#     y: Annotated[float, Field(strict=True, gt=0)] = 3.98
#     z: Annotated[float, Field(strict=True, gt=0)] = 2.43


class DiamondRaw(BaseModel):
    # Price is 326
    carat: Annotated[float, Field(strict=True, gt=0)] = 0.23
