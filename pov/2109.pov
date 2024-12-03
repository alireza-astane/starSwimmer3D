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
    sphere { m*<1.1946520381380399,0.14793692406034567,0.5722253911789549>, 1 }        
    sphere {  m*<1.4388695271010319,0.1589547738914475,3.562247723460697>, 1 }
    sphere {  m*<3.9321167161635695,0.1589547738914475,-0.6550344850299217>, 1 }
    sphere {  m*<-3.433668283808584,7.650117663648167,-2.1643444387037105>, 1}
    sphere { m*<-3.729889486528465,-8.053582097945092,-2.338804298936653>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.4388695271010319,0.1589547738914475,3.562247723460697>, <1.1946520381380399,0.14793692406034567,0.5722253911789549>, 0.5 }
    cylinder { m*<3.9321167161635695,0.1589547738914475,-0.6550344850299217>, <1.1946520381380399,0.14793692406034567,0.5722253911789549>, 0.5}
    cylinder { m*<-3.433668283808584,7.650117663648167,-2.1643444387037105>, <1.1946520381380399,0.14793692406034567,0.5722253911789549>, 0.5 }
    cylinder {  m*<-3.729889486528465,-8.053582097945092,-2.338804298936653>, <1.1946520381380399,0.14793692406034567,0.5722253911789549>, 0.5}

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
    sphere { m*<1.1946520381380399,0.14793692406034567,0.5722253911789549>, 1 }        
    sphere {  m*<1.4388695271010319,0.1589547738914475,3.562247723460697>, 1 }
    sphere {  m*<3.9321167161635695,0.1589547738914475,-0.6550344850299217>, 1 }
    sphere {  m*<-3.433668283808584,7.650117663648167,-2.1643444387037105>, 1}
    sphere { m*<-3.729889486528465,-8.053582097945092,-2.338804298936653>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.4388695271010319,0.1589547738914475,3.562247723460697>, <1.1946520381380399,0.14793692406034567,0.5722253911789549>, 0.5 }
    cylinder { m*<3.9321167161635695,0.1589547738914475,-0.6550344850299217>, <1.1946520381380399,0.14793692406034567,0.5722253911789549>, 0.5}
    cylinder { m*<-3.433668283808584,7.650117663648167,-2.1643444387037105>, <1.1946520381380399,0.14793692406034567,0.5722253911789549>, 0.5 }
    cylinder {  m*<-3.729889486528465,-8.053582097945092,-2.338804298936653>, <1.1946520381380399,0.14793692406034567,0.5722253911789549>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    