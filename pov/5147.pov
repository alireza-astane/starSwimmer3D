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
    sphere { m*<-0.439249170508548,-0.14494676290290495,-1.605991388997505>, 1 }        
    sphere {  m*<0.46349401491298187,0.2892311748338572,8.343704289946688>, 1 }
    sphere {  m*<3.22311657369156,-0.010020119889576767,-3.2966138137278396>, 1 }
    sphere {  m*<-2.0696787032724986,2.1836280864577557,-2.5648629723920866>, 1}
    sphere { m*<-1.801891482234667,-2.7040638559461416,-2.375316687229516>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.46349401491298187,0.2892311748338572,8.343704289946688>, <-0.439249170508548,-0.14494676290290495,-1.605991388997505>, 0.5 }
    cylinder { m*<3.22311657369156,-0.010020119889576767,-3.2966138137278396>, <-0.439249170508548,-0.14494676290290495,-1.605991388997505>, 0.5}
    cylinder { m*<-2.0696787032724986,2.1836280864577557,-2.5648629723920866>, <-0.439249170508548,-0.14494676290290495,-1.605991388997505>, 0.5 }
    cylinder {  m*<-1.801891482234667,-2.7040638559461416,-2.375316687229516>, <-0.439249170508548,-0.14494676290290495,-1.605991388997505>, 0.5}

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
    sphere { m*<-0.439249170508548,-0.14494676290290495,-1.605991388997505>, 1 }        
    sphere {  m*<0.46349401491298187,0.2892311748338572,8.343704289946688>, 1 }
    sphere {  m*<3.22311657369156,-0.010020119889576767,-3.2966138137278396>, 1 }
    sphere {  m*<-2.0696787032724986,2.1836280864577557,-2.5648629723920866>, 1}
    sphere { m*<-1.801891482234667,-2.7040638559461416,-2.375316687229516>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.46349401491298187,0.2892311748338572,8.343704289946688>, <-0.439249170508548,-0.14494676290290495,-1.605991388997505>, 0.5 }
    cylinder { m*<3.22311657369156,-0.010020119889576767,-3.2966138137278396>, <-0.439249170508548,-0.14494676290290495,-1.605991388997505>, 0.5}
    cylinder { m*<-2.0696787032724986,2.1836280864577557,-2.5648629723920866>, <-0.439249170508548,-0.14494676290290495,-1.605991388997505>, 0.5 }
    cylinder {  m*<-1.801891482234667,-2.7040638559461416,-2.375316687229516>, <-0.439249170508548,-0.14494676290290495,-1.605991388997505>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    