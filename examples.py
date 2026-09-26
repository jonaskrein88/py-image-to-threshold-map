
from imageToDither import convert_image, DitherPattern
from pathlib import Path



DIR = Path(__file__).resolve().parent

preview_scale = 4

convert_image(DIR / "gengar.png", noiseAmount=0.75, finalUpscale=preview_scale)
convert_image(DIR / "swirls.jpg", resize=(64,64), noiseAmount=0.5, noiseType = DitherPattern.BAYER2X2, finalUpscale=preview_scale)
convert_image(DIR / "geoTest.png", resize=(64,64), noiseAmount = 0.75, noiseType = DitherPattern.BLUE_NOISE, finalUpscale=preview_scale)

convert_image(DIR / "bubbleTiles.png", resize=(64,64), noiseType=DitherPattern.INTERLEAVED_GRADIENT_NOISE, noiseAmount=0.6, finalUpscale=preview_scale)
convert_image(DIR / "Gundam_Wing_logo_full.png", noiseType=DitherPattern.BAYER4X4, noiseAmount=0.1, resize=(128,64), finalUpscale=preview_scale)

