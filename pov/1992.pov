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
    sphere { m*<1.272005371381602,-7.360169249977675e-19,0.6267062708139405>, 1 }        
    sphere {  m*<1.5142182143342298,-1.147319532231418e-19,3.6169225801828526>, 1 }
    sphere {  m*<4.055102200509661,5.891696816341801e-18,-0.6186989424618397>, 1 }
    sphere {  m*<-3.6957916309421384,8.164965809277259,-2.315155531279795>, 1}
    sphere { m*<-3.6957916309421384,-8.164965809277259,-2.3151555312797987>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.5142182143342298,-1.147319532231418e-19,3.6169225801828526>, <1.272005371381602,-7.360169249977675e-19,0.6267062708139405>, 0.5 }
    cylinder { m*<4.055102200509661,5.891696816341801e-18,-0.6186989424618397>, <1.272005371381602,-7.360169249977675e-19,0.6267062708139405>, 0.5}
    cylinder { m*<-3.6957916309421384,8.164965809277259,-2.315155531279795>, <1.272005371381602,-7.360169249977675e-19,0.6267062708139405>, 0.5 }
    cylinder {  m*<-3.6957916309421384,-8.164965809277259,-2.3151555312797987>, <1.272005371381602,-7.360169249977675e-19,0.6267062708139405>, 0.5}

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
    sphere { m*<1.272005371381602,-7.360169249977675e-19,0.6267062708139405>, 1 }        
    sphere {  m*<1.5142182143342298,-1.147319532231418e-19,3.6169225801828526>, 1 }
    sphere {  m*<4.055102200509661,5.891696816341801e-18,-0.6186989424618397>, 1 }
    sphere {  m*<-3.6957916309421384,8.164965809277259,-2.315155531279795>, 1}
    sphere { m*<-3.6957916309421384,-8.164965809277259,-2.3151555312797987>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.5142182143342298,-1.147319532231418e-19,3.6169225801828526>, <1.272005371381602,-7.360169249977675e-19,0.6267062708139405>, 0.5 }
    cylinder { m*<4.055102200509661,5.891696816341801e-18,-0.6186989424618397>, <1.272005371381602,-7.360169249977675e-19,0.6267062708139405>, 0.5}
    cylinder { m*<-3.6957916309421384,8.164965809277259,-2.315155531279795>, <1.272005371381602,-7.360169249977675e-19,0.6267062708139405>, 0.5 }
    cylinder {  m*<-3.6957916309421384,-8.164965809277259,-2.3151555312797987>, <1.272005371381602,-7.360169249977675e-19,0.6267062708139405>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    