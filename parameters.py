from enum import Enum


class Strain(Enum):
    chlorella_vulgaris = "Chlorella vulgaris"
    nannochloropsis_oceanica = "Nannochloropsis oceanica"
    desmodesmus_intermedius = "Desmodesmus intermedius"


class Period(Enum):
    summer = "Summer"
    fall = "Fall"
    winter = "Winter"
    spring = "Spring"
    annual_average = "Annual Average"
    custom = "Custom"
