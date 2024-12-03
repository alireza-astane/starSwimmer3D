#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<1.1931067274013911,0.15060053104348453,0.5713116942906393>, 1 }        
    sphere {  m*<1.4373230777163943,0.1618243467470623,3.561333344679067>, 1 }
    sphere {  m*<3.9305702667789317,0.1618243467470623,-0.6559488638115512>, 1 }
    sphere {  m*<-3.42896068883253,7.640763084637968,-2.1615609233961024>, 1}
    sphere { m*<-3.7305952661128594,-8.051598838617943,-2.3392216381807263>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.4373230777163943,0.1618243467470623,3.561333344679067>, <1.1931067274013911,0.15060053104348453,0.5713116942906393>, 0.5 }
    cylinder { m*<3.9305702667789317,0.1618243467470623,-0.6559488638115512>, <1.1931067274013911,0.15060053104348453,0.5713116942906393>, 0.5}
    cylinder { m*<-3.42896068883253,7.640763084637968,-2.1615609233961024>, <1.1931067274013911,0.15060053104348453,0.5713116942906393>, 0.5 }
    cylinder {  m*<-3.7305952661128594,-8.051598838617943,-2.3392216381807263>, <1.1931067274013911,0.15060053104348453,0.5713116942906393>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<1.1931067274013911,0.15060053104348453,0.5713116942906393>, 1 }        
    sphere {  m*<1.4373230777163943,0.1618243467470623,3.561333344679067>, 1 }
    sphere {  m*<3.9305702667789317,0.1618243467470623,-0.6559488638115512>, 1 }
    sphere {  m*<-3.42896068883253,7.640763084637968,-2.1615609233961024>, 1}
    sphere { m*<-3.7305952661128594,-8.051598838617943,-2.3392216381807263>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.4373230777163943,0.1618243467470623,3.561333344679067>, <1.1931067274013911,0.15060053104348453,0.5713116942906393>, 0.5 }
    cylinder { m*<3.9305702667789317,0.1618243467470623,-0.6559488638115512>, <1.1931067274013911,0.15060053104348453,0.5713116942906393>, 0.5}
    cylinder { m*<-3.42896068883253,7.640763084637968,-2.1615609233961024>, <1.1931067274013911,0.15060053104348453,0.5713116942906393>, 0.5 }
    cylinder {  m*<-3.7305952661128594,-8.051598838617943,-2.3392216381807263>, <1.1931067274013911,0.15060053104348453,0.5713116942906393>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    