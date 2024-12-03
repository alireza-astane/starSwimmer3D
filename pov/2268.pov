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
    sphere { m*<1.0711456651691613,0.35640959280968376,0.4991998843205085>, 1 }        
    sphere {  m*<1.3151982738118764,0.38442489832148596,3.4891240195212223>, 1 }
    sphere {  m*<3.80844546287441,0.38442489832148585,-0.7281581889693955>, 1 }
    sphere {  m*<-3.0546499203054727,6.907181454796789,-1.9402382054145495>, 1}
    sphere { m*<-3.783876688492201,-7.900867159651322,-2.3707278749998366>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3151982738118764,0.38442489832148596,3.4891240195212223>, <1.0711456651691613,0.35640959280968376,0.4991998843205085>, 0.5 }
    cylinder { m*<3.80844546287441,0.38442489832148585,-0.7281581889693955>, <1.0711456651691613,0.35640959280968376,0.4991998843205085>, 0.5}
    cylinder { m*<-3.0546499203054727,6.907181454796789,-1.9402382054145495>, <1.0711456651691613,0.35640959280968376,0.4991998843205085>, 0.5 }
    cylinder {  m*<-3.783876688492201,-7.900867159651322,-2.3707278749998366>, <1.0711456651691613,0.35640959280968376,0.4991998843205085>, 0.5}

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
    sphere { m*<1.0711456651691613,0.35640959280968376,0.4991998843205085>, 1 }        
    sphere {  m*<1.3151982738118764,0.38442489832148596,3.4891240195212223>, 1 }
    sphere {  m*<3.80844546287441,0.38442489832148585,-0.7281581889693955>, 1 }
    sphere {  m*<-3.0546499203054727,6.907181454796789,-1.9402382054145495>, 1}
    sphere { m*<-3.783876688492201,-7.900867159651322,-2.3707278749998366>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3151982738118764,0.38442489832148596,3.4891240195212223>, <1.0711456651691613,0.35640959280968376,0.4991998843205085>, 0.5 }
    cylinder { m*<3.80844546287441,0.38442489832148585,-0.7281581889693955>, <1.0711456651691613,0.35640959280968376,0.4991998843205085>, 0.5}
    cylinder { m*<-3.0546499203054727,6.907181454796789,-1.9402382054145495>, <1.0711456651691613,0.35640959280968376,0.4991998843205085>, 0.5 }
    cylinder {  m*<-3.783876688492201,-7.900867159651322,-2.3707278749998366>, <1.0711456651691613,0.35640959280968376,0.4991998843205085>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    