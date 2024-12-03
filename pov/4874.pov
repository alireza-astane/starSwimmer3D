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
    sphere { m*<-0.250680431360758,-0.12924518972889598,-1.4666784855414627>, 1 }        
    sphere {  m*<0.48155551222855286,0.26224796473625944,7.620467276616869>, 1 }
    sphere {  m*<2.484027962645499,-0.027211214342521807,-2.6958880109926446>, 1 }
    sphere {  m*<-1.872295791253648,2.1992287546897034,-2.440624250957431>, 1}
    sphere { m*<-1.6045085702158162,-2.688463187714194,-2.2510779657948583>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.48155551222855286,0.26224796473625944,7.620467276616869>, <-0.250680431360758,-0.12924518972889598,-1.4666784855414627>, 0.5 }
    cylinder { m*<2.484027962645499,-0.027211214342521807,-2.6958880109926446>, <-0.250680431360758,-0.12924518972889598,-1.4666784855414627>, 0.5}
    cylinder { m*<-1.872295791253648,2.1992287546897034,-2.440624250957431>, <-0.250680431360758,-0.12924518972889598,-1.4666784855414627>, 0.5 }
    cylinder {  m*<-1.6045085702158162,-2.688463187714194,-2.2510779657948583>, <-0.250680431360758,-0.12924518972889598,-1.4666784855414627>, 0.5}

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
    sphere { m*<-0.250680431360758,-0.12924518972889598,-1.4666784855414627>, 1 }        
    sphere {  m*<0.48155551222855286,0.26224796473625944,7.620467276616869>, 1 }
    sphere {  m*<2.484027962645499,-0.027211214342521807,-2.6958880109926446>, 1 }
    sphere {  m*<-1.872295791253648,2.1992287546897034,-2.440624250957431>, 1}
    sphere { m*<-1.6045085702158162,-2.688463187714194,-2.2510779657948583>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.48155551222855286,0.26224796473625944,7.620467276616869>, <-0.250680431360758,-0.12924518972889598,-1.4666784855414627>, 0.5 }
    cylinder { m*<2.484027962645499,-0.027211214342521807,-2.6958880109926446>, <-0.250680431360758,-0.12924518972889598,-1.4666784855414627>, 0.5}
    cylinder { m*<-1.872295791253648,2.1992287546897034,-2.440624250957431>, <-0.250680431360758,-0.12924518972889598,-1.4666784855414627>, 0.5 }
    cylinder {  m*<-1.6045085702158162,-2.688463187714194,-2.2510779657948583>, <-0.250680431360758,-0.12924518972889598,-1.4666784855414627>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    