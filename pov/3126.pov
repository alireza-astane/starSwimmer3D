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
    sphere { m*<0.416242883540127,0.9947258970479909,0.11311804285031216>, 1 }        
    sphere {  m*<0.6569779882818187,1.1234359752283165,3.1006728139708635>, 1 }
    sphere {  m*<3.1509512775463833,1.0967598724343655,-1.116091482600872>, 1 }
    sphere {  m*<-1.2053724763527627,3.323199841466592,-0.8608277225656581>, 1}
    sphere { m*<-3.695333174162686,-6.777619162833361,-2.2691031786792455>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6569779882818187,1.1234359752283165,3.1006728139708635>, <0.416242883540127,0.9947258970479909,0.11311804285031216>, 0.5 }
    cylinder { m*<3.1509512775463833,1.0967598724343655,-1.116091482600872>, <0.416242883540127,0.9947258970479909,0.11311804285031216>, 0.5}
    cylinder { m*<-1.2053724763527627,3.323199841466592,-0.8608277225656581>, <0.416242883540127,0.9947258970479909,0.11311804285031216>, 0.5 }
    cylinder {  m*<-3.695333174162686,-6.777619162833361,-2.2691031786792455>, <0.416242883540127,0.9947258970479909,0.11311804285031216>, 0.5}

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
    sphere { m*<0.416242883540127,0.9947258970479909,0.11311804285031216>, 1 }        
    sphere {  m*<0.6569779882818187,1.1234359752283165,3.1006728139708635>, 1 }
    sphere {  m*<3.1509512775463833,1.0967598724343655,-1.116091482600872>, 1 }
    sphere {  m*<-1.2053724763527627,3.323199841466592,-0.8608277225656581>, 1}
    sphere { m*<-3.695333174162686,-6.777619162833361,-2.2691031786792455>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6569779882818187,1.1234359752283165,3.1006728139708635>, <0.416242883540127,0.9947258970479909,0.11311804285031216>, 0.5 }
    cylinder { m*<3.1509512775463833,1.0967598724343655,-1.116091482600872>, <0.416242883540127,0.9947258970479909,0.11311804285031216>, 0.5}
    cylinder { m*<-1.2053724763527627,3.323199841466592,-0.8608277225656581>, <0.416242883540127,0.9947258970479909,0.11311804285031216>, 0.5 }
    cylinder {  m*<-3.695333174162686,-6.777619162833361,-2.2691031786792455>, <0.416242883540127,0.9947258970479909,0.11311804285031216>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    