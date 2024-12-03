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
    sphere { m*<-1.3682299554102566,-0.5484452150418176,-0.9109436178766743>, 1 }        
    sphere {  m*<0.08493759105160592,0.0651204165715416,8.963947103290117>, 1 }
    sphere {  m*<7.440289029051576,-0.023799859422815733,-5.615546186755237>, 1 }
    sphere {  m*<-4.484269400956736,3.490119597862781,-2.507732653471798>, 1}
    sphere { m*<-2.697676050126014,-3.156535144091912,-1.565740137419822>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.08493759105160592,0.0651204165715416,8.963947103290117>, <-1.3682299554102566,-0.5484452150418176,-0.9109436178766743>, 0.5 }
    cylinder { m*<7.440289029051576,-0.023799859422815733,-5.615546186755237>, <-1.3682299554102566,-0.5484452150418176,-0.9109436178766743>, 0.5}
    cylinder { m*<-4.484269400956736,3.490119597862781,-2.507732653471798>, <-1.3682299554102566,-0.5484452150418176,-0.9109436178766743>, 0.5 }
    cylinder {  m*<-2.697676050126014,-3.156535144091912,-1.565740137419822>, <-1.3682299554102566,-0.5484452150418176,-0.9109436178766743>, 0.5}

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
    sphere { m*<-1.3682299554102566,-0.5484452150418176,-0.9109436178766743>, 1 }        
    sphere {  m*<0.08493759105160592,0.0651204165715416,8.963947103290117>, 1 }
    sphere {  m*<7.440289029051576,-0.023799859422815733,-5.615546186755237>, 1 }
    sphere {  m*<-4.484269400956736,3.490119597862781,-2.507732653471798>, 1}
    sphere { m*<-2.697676050126014,-3.156535144091912,-1.565740137419822>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.08493759105160592,0.0651204165715416,8.963947103290117>, <-1.3682299554102566,-0.5484452150418176,-0.9109436178766743>, 0.5 }
    cylinder { m*<7.440289029051576,-0.023799859422815733,-5.615546186755237>, <-1.3682299554102566,-0.5484452150418176,-0.9109436178766743>, 0.5}
    cylinder { m*<-4.484269400956736,3.490119597862781,-2.507732653471798>, <-1.3682299554102566,-0.5484452150418176,-0.9109436178766743>, 0.5 }
    cylinder {  m*<-2.697676050126014,-3.156535144091912,-1.565740137419822>, <-1.3682299554102566,-0.5484452150418176,-0.9109436178766743>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    