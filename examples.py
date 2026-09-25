
from imageToDither import convert_image, DitherPattern
from pathlib import Path



DIR = Path(__file__).resolve().parent

convert_image(DIR / "gengar.png", noiseAmount=0.2)
convert_image(DIR / "geoTest.png", resize=(32,32), noiseAmount = 0.75, noiseType = 'blueNoise')

convert_image(DIR / "bubbleTiles.png", noiseType='ign', noiseAmount=0.6, resize=(32,32))
convert_image(DIR / "Gundam_Wing_logo_full.png", noiseType=DitherPattern.BAYER4X4, noiseAmount=0.1, resize=(128,64))

