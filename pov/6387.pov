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
    sphere { m*<-1.32816853037833,-0.6070064242038167,-0.890373894358807>, 1 }        
    sphere {  m*<0.12260236888441889,0.03374029766440198,8.983141563976918>, 1 }
    sphere {  m*<7.477953806884387,-0.055179978329955276,-5.596351726068429>, 1 }
    sphere {  m*<-4.6787255559774525,3.694136109594597,-2.6070867411965866>, 1}
    sphere { m*<-2.646596242055947,-3.2220812532979526,-1.5395687160069609>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.12260236888441889,0.03374029766440198,8.983141563976918>, <-1.32816853037833,-0.6070064242038167,-0.890373894358807>, 0.5 }
    cylinder { m*<7.477953806884387,-0.055179978329955276,-5.596351726068429>, <-1.32816853037833,-0.6070064242038167,-0.890373894358807>, 0.5}
    cylinder { m*<-4.6787255559774525,3.694136109594597,-2.6070867411965866>, <-1.32816853037833,-0.6070064242038167,-0.890373894358807>, 0.5 }
    cylinder {  m*<-2.646596242055947,-3.2220812532979526,-1.5395687160069609>, <-1.32816853037833,-0.6070064242038167,-0.890373894358807>, 0.5}

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
    sphere { m*<-1.32816853037833,-0.6070064242038167,-0.890373894358807>, 1 }        
    sphere {  m*<0.12260236888441889,0.03374029766440198,8.983141563976918>, 1 }
    sphere {  m*<7.477953806884387,-0.055179978329955276,-5.596351726068429>, 1 }
    sphere {  m*<-4.6787255559774525,3.694136109594597,-2.6070867411965866>, 1}
    sphere { m*<-2.646596242055947,-3.2220812532979526,-1.5395687160069609>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.12260236888441889,0.03374029766440198,8.983141563976918>, <-1.32816853037833,-0.6070064242038167,-0.890373894358807>, 0.5 }
    cylinder { m*<7.477953806884387,-0.055179978329955276,-5.596351726068429>, <-1.32816853037833,-0.6070064242038167,-0.890373894358807>, 0.5}
    cylinder { m*<-4.6787255559774525,3.694136109594597,-2.6070867411965866>, <-1.32816853037833,-0.6070064242038167,-0.890373894358807>, 0.5 }
    cylinder {  m*<-2.646596242055947,-3.2220812532979526,-1.5395687160069609>, <-1.32816853037833,-0.6070064242038167,-0.890373894358807>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    