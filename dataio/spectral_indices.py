import numpy as np


class spectral_indices:
    

    @staticmethod
    def normalized_difference(img, bands):
        '''
            Calculates the normalized difference betweeen two bands.

            Inputs:
                - img: is a multispectral/iperspectral image. Must be a numpy
                       array (W,H,B) width W, height H and B bands
                - bands: a tuple of two elements corresponing to the indexes of the
                         bands to be used for the normalized_difference
            Output:
                - nd: the normalized difference of bands of img. See formula below. 
            ------------------------------------------------------------------------
            The formula applied is the following

                                nd = (b1 - b2)/((b1 + b2) + e)

            e ~= 0.00001 to stabilize division with weak denominator.
        '''
        SMALL_VALUE = 1
        # Select image bands
        b1 = img[:,:,bands[0]]
        b2 = img[:,:,bands[1]]

        # Check for NaN and remove them
        b1[np.isnan(b1)] = 0
        b2[np.isnan(b2)] = 0

        # Apply formula
        nd = (b1 - b2)/ ((b1 + b2) + SMALL_VALUE)

        return nd

    @staticmethod
    def weighted_normalized_difference(img, bands, a):
        '''
            Calculates the weighted normalized difference betweeen three bands.
            Inputs:
                - img: is a multispectral/iperspectral image. Must be a numpy
                       array (W,H,B) width W, height H and B bands
                - bands: a tuple of three elements corresponing to the indexes of the
                         bands to be used for the weighted_normalized_difference
                - a: is a weighted coefficient ∈ [0:1]
            Output:
                - wnd: the weighted normalized difference of bands of img. See formula below. 
            ------------------------------------------------------------------------
            The formula applied is the following
                                nd = (b1 - a*b2 - (1-a)*b3)/((b1 + a*b2 + (1-a)*b3) + e)
            e ~= 0.00001 to stabilize division with weak denominator.
        '''
        SMALL_VALUE = 1
        # Select image bands
        b1 = img[:,:,bands[0]]
        b2 = img[:,:,bands[1]]
        b3 = img[:,:,bands[2]]

        # Check for NaN and remove them
        b1[np.isnan(b1)] = 0
        b2[np.isnan(b2)] = 0
        b3[np.isnan(b3)] = 0


        # Apply formula
        wnd = (b1 - a*b2 - (1-a)*b3)/ ((b1 + a*b2 + (1-a)*b3) + SMALL_VALUE)

        return wnd

    @staticmethod
    def compute_NDDI(NDWI,NDVI):
        '''
            Calculates the normalized difference betweeen two indices.

            Inputs:
                - NDWI: NDWI image as computed by normalized_difference method. Must be a numpy
                       array (W,H,B) width W, height H and B bands
                - NDVI: NDVI image as computed by normalized_difference method. Must be a numpy
                       array (W,H,B) width W, height H and B bands
            Output:
                - NDDI: normalized difference of NDWI and NDVI bands 
            ------------------------------------------------------------------------
            The formula applied is the following

                                nd = (NDVI - NDWI)/((NDVI + NDWI) + e)

            e ~= 0.00001 to stabilize division with weak denominator.
        '''
        # NDDI = (NDVI-NDWI)/(NDVI+NDWI+0.00001)
        # print(NDDI)
        # globMax = 0
        # for row in NDDI:
        #     maxRow = max(abs(row))
        #     if maxRow > globMax:
        #         globMax = maxRow
        # NDDI /= globMax
        # return NDDI
        SMALL_VALUE = 1
        return (NDVI - NDWI) / (NDVI + NDWI + SMALL_VALUE)


