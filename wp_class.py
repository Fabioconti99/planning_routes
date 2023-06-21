import os 

# lat = y
# lon = x

class wp:

    def __init__(self,real_lat = 0.0,real_lon = 0.0,pix_lat = 0.0,pix_lon = 0.0): # contructor method

        self.real_lat = real_lat
        self.real_lon = real_lon

        self.pix_lat = pix_lat
        self.pix_lon = pix_lon

    def set_pix(self, pix_lat, pix_lon):

        self.__pix_lat = pix_lat
        self.__pix_lon = pix_lon

    def set_real(self,real_lat,real_lon):

        self.__real_lat = real_lat
        self.__real_lon = real_lon

    def get_pix(self):

        return self.__pix_lat, self.__pix_lon

    def get_real(self):

        return self.__real_lat, self.__real_lon







