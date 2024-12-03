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
    sphere { m*<0.36064903965112066,-5.138205494611212e-18,1.063689149071068>, 1 }        
    sphere {  m*<0.4110633698300554,-2.9303985323683875e-18,4.063267152326999>, 1 }
    sphere {  m*<8.025884867301775,2.7607540753513513e-18,-1.7922956185974>, 1 }
    sphere {  m*<-4.4087489867705285,8.164965809277259,-2.189956821794409>, 1}
    sphere { m*<-4.4087489867705285,-8.164965809277259,-2.1899568217944125>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4110633698300554,-2.9303985323683875e-18,4.063267152326999>, <0.36064903965112066,-5.138205494611212e-18,1.063689149071068>, 0.5 }
    cylinder { m*<8.025884867301775,2.7607540753513513e-18,-1.7922956185974>, <0.36064903965112066,-5.138205494611212e-18,1.063689149071068>, 0.5}
    cylinder { m*<-4.4087489867705285,8.164965809277259,-2.189956821794409>, <0.36064903965112066,-5.138205494611212e-18,1.063689149071068>, 0.5 }
    cylinder {  m*<-4.4087489867705285,-8.164965809277259,-2.1899568217944125>, <0.36064903965112066,-5.138205494611212e-18,1.063689149071068>, 0.5}

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
    sphere { m*<0.36064903965112066,-5.138205494611212e-18,1.063689149071068>, 1 }        
    sphere {  m*<0.4110633698300554,-2.9303985323683875e-18,4.063267152326999>, 1 }
    sphere {  m*<8.025884867301775,2.7607540753513513e-18,-1.7922956185974>, 1 }
    sphere {  m*<-4.4087489867705285,8.164965809277259,-2.189956821794409>, 1}
    sphere { m*<-4.4087489867705285,-8.164965809277259,-2.1899568217944125>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4110633698300554,-2.9303985323683875e-18,4.063267152326999>, <0.36064903965112066,-5.138205494611212e-18,1.063689149071068>, 0.5 }
    cylinder { m*<8.025884867301775,2.7607540753513513e-18,-1.7922956185974>, <0.36064903965112066,-5.138205494611212e-18,1.063689149071068>, 0.5}
    cylinder { m*<-4.4087489867705285,8.164965809277259,-2.189956821794409>, <0.36064903965112066,-5.138205494611212e-18,1.063689149071068>, 0.5 }
    cylinder {  m*<-4.4087489867705285,-8.164965809277259,-2.1899568217944125>, <0.36064903965112066,-5.138205494611212e-18,1.063689149071068>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    