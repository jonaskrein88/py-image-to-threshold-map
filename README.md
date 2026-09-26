# py-image-to-threshold-map
Converts input bitmaps to ordered threshold maps for dithering.

![Example](images/gengar_dbg_inputScaled.png) ![Example](images/gengar_dbg_noise.png) ![Example](images/gengar_dbg_final.png)
![Example](images/bubbleTiles_dbg_inputScaled.png) ![Example](images/bubbleTiles_dbg_noise.png) ![Example](images/bubbleTiles_dbg_final.png)
![Example](images/geoTest_dbg_inputScaled.png) ![Example](images/geoTest_dbg_noise.png) ![Example](images/geoTest_dbg_final.png)

![Example](images/Swirls_dbg_inputScaled.png) ![Example](images/Swirls_out.png)

![Example](images/geoTest.png) ![Example](images/geoTest_out.png)

![Example](images/Gundam_Wing_logo_full.png) ![Example](images/Gundam_Wing_logo_full_out.png)

Example of the threshold maps being applied in a [Unity post process shader](https://www.artstation.com/artwork/dyA5g3)
![Example](images/dithering_unity.png)

Blue Noise function from Bart Wronski:
https://bartwronski.com/2022/08/31/progressive-image-stippling-and-greedy-blue-noise-importance-sampling/
