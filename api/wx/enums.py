from enum import Enum


class OperationEnum(Enum):
    NONE = '', 'None'
    AVERAGE = 'AVG', 'Average'
    MINIMUM = 'MIN', 'Minimum'
    MAXIMUM = 'MAX', 'Maximum'
    MEDIAN = 'MED', 'Median'
    STANDARD_DEVIATION = 'STD', 'Standard deviation'
    COUNT = 'COUNT', 'Count'
    SUM = 'SUM', 'Sum'

    def __init__(self, symbol, name):
        self.symbol = symbol
        self.name_ = name


class QualityFlagEnum(Enum):
    NOT_CHECKED = '-', 'Not checked', 1
    SUSPICIOUS = 'S', 'Suspicious', 2
    BAD = 'B', 'Bad', 3
    GOOD = 'G', 'Good', 4

    def __init__(self, symbol, name, id):
        self.symbol = symbol
        self.name_ = name
        self.id = id


class QuantityEnum(Enum):
    DIMENSIONLESS = 'DIMENSIONLESS', 'Dimensionless'
    LENGTH_WIDTH_HEIGHT_DEPTH = 'LENGTH/WIDTH/HEIGHT/DEPTH', 'Length, width, height, or depth'
    TEMPERATURE = 'TEMPERATURE', 'Temperature'
    ANGLE = 'ANGLE', 'Angle'
    ELECTRIC_POTENTIAL_DIFFERENCE = 'ELECTRIC_POTENTIAL_DIFFERENCE', 'Electric potential difference'
    SPEED = 'SPEED', 'Speed'
    PRESSURE = 'PRESSURE', 'Pressure'
    RADIATION = 'RADIATION', 'Radiation'
    DURATION = 'DURATION', 'Duration'
    LINEAR_LENGTH_WIDTH_HEIGHT_DEPTH_RATE = 'LINEAR_L/W/H/D_RATE', 'Linear length/width/height/depth rate'

    def __init__(self, symbol, name):
        self.symbol = symbol
        self.name_ = name


class ElementEnum(Enum):
    TEMPERATURE = ('T', 'Air temperature', QuantityEnum.TEMPERATURE)
    DEWPOINT = ('TD', 'Dewpoint temperature', QuantityEnum.TEMPERATURE)
    HUMIDITY = ('RH', 'Relative humidity', QuantityEnum.DIMENSIONLESS)
    ATMOSPHERIC_PRESSURE = ('ATMPRESS', 'Atmospheric pressure', QuantityEnum.PRESSURE)
    RAIN_AMOUNT = ('RAIN', 'Rainfall', QuantityEnum.LENGTH_WIDTH_HEIGHT_DEPTH)
    # RAINFALL_INTENSITY = ('RAINFALL_RATE', 'Rainfall rate', QuantityEnum.LINEAR_LENGTH_WIDTH_HEIGHT_DEPTH_RATE)
    # SNOWFALL_INTENSITY = ('SNOWFALL_RATE', 'Snowfall rate', QuantityEnum.LINEAR_LENGTH_WIDTH_HEIGHT_DEPTH_RATE)
    WIND_SPEED = ('WSPD', 'Wind speed', QuantityEnum.SPEED)
    WIND_DIRECTION = ('WDIR', 'Wind direction', QuantityEnum.ANGLE)
    SOLAR_RADIATION = ('SOLRAD', 'Solar radiation', QuantityEnum.RADIATION)
    WATER_LEVEL = ('WTLVL', 'Water level', QuantityEnum.LENGTH_WIDTH_HEIGHT_DEPTH)
    SOLAR_PANEL_VOLTAGE = ('SLRPANEL', 'Solar panel voltage', QuantityEnum.ELECTRIC_POTENTIAL_DIFFERENCE)
    BATTERY_VOLTAGE = ('BATTERY', 'Battery voltage', QuantityEnum.ELECTRIC_POTENTIAL_DIFFERENCE)

    def __init__(self, symbol, name, quantity):
        self.symbol = symbol
        self.name_ = name
        self.quantity = quantity


class UnitEnum(Enum):
    PERCENTAGE = '%', 'percentage', QuantityEnum.DIMENSIONLESS
    CELSIUS = 'C', 'Celsius degrees', QuantityEnum.TEMPERATURE
    FAHRENHEIT = 'F', 'Fahrenheit', QuantityEnum.TEMPERATURE
    METERS = 'm', 'meters', QuantityEnum.LENGTH_WIDTH_HEIGHT_DEPTH
    CENTIMETERS = 'cm', 'centimeters', QuantityEnum.LENGTH_WIDTH_HEIGHT_DEPTH
    MILLIMETERS = 'mm', 'millimeters', QuantityEnum.LENGTH_WIDTH_HEIGHT_DEPTH
    INCHES = 'in', 'inches', QuantityEnum.LENGTH_WIDTH_HEIGHT_DEPTH
    MILIMETERS_PER_HOUR = 'mm/h', 'millimeters per hour', QuantityEnum.LINEAR_LENGTH_WIDTH_HEIGHT_DEPTH_RATE
    INCHES_PER_HOUR = 'in/h', 'inches per hour', QuantityEnum.LINEAR_LENGTH_WIDTH_HEIGHT_DEPTH_RATE
    WATTS_PER_SQUARE_METER = "W/m²", 'watts per square meter', QuantityEnum.RADIATION
    MILLIWATTS_PER_SQUARE_CENTIMETER = "mW/cm²", 'milliwatts per square centimeter', QuantityEnum.RADIATION
    METERS_PER_SECOND = "m/s", 'meters per second', QuantityEnum.SPEED
    MILES_PER_HOUR = "mph", 'miles per hour', QuantityEnum.SPEED
    KNOTS = "kn", 'knots', QuantityEnum.SPEED
    DEGREES = "°", 'degrees', QuantityEnum.ANGLE
    VOLTS = 'V', 'Volts', QuantityEnum.ELECTRIC_POTENTIAL_DIFFERENCE
    PASCALS = 'Pa', 'Pascals', QuantityEnum.PRESSURE
    HECTO_PASCALS = 'hPa', 'hecto Pascals', QuantityEnum.PRESSURE
    INCHES_OF_MERCURY = 'inHg', 'inches of mercury', QuantityEnum.PRESSURE

    def __init__(self, symbol, name, quantity):
        self.symbol = symbol
        self.name_ = name
        self.quantity = quantity


class FlashTypeEnum(Enum):
    CG = 'CG'
    IC = 'IC'
