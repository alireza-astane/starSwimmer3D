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
    sphere { m*<0.3878763662811349,-5.6313500753238994e-18,1.0530649992515728>, 1 }        
    sphere {  m*<0.44255231990211974,-2.905001727934062e-18,4.052568499960156>, 1 }
    sphere {  m*<7.917923627050284,2.4264882103523587e-18,-1.764750498535746>, 1 }
    sphere {  m*<-4.386139727839583,8.164965809277259,-2.1938016958174353>, 1}
    sphere { m*<-4.386139727839583,-8.164965809277259,-2.193801695817439>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.44255231990211974,-2.905001727934062e-18,4.052568499960156>, <0.3878763662811349,-5.6313500753238994e-18,1.0530649992515728>, 0.5 }
    cylinder { m*<7.917923627050284,2.4264882103523587e-18,-1.764750498535746>, <0.3878763662811349,-5.6313500753238994e-18,1.0530649992515728>, 0.5}
    cylinder { m*<-4.386139727839583,8.164965809277259,-2.1938016958174353>, <0.3878763662811349,-5.6313500753238994e-18,1.0530649992515728>, 0.5 }
    cylinder {  m*<-4.386139727839583,-8.164965809277259,-2.193801695817439>, <0.3878763662811349,-5.6313500753238994e-18,1.0530649992515728>, 0.5}

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
    sphere { m*<0.3878763662811349,-5.6313500753238994e-18,1.0530649992515728>, 1 }        
    sphere {  m*<0.44255231990211974,-2.905001727934062e-18,4.052568499960156>, 1 }
    sphere {  m*<7.917923627050284,2.4264882103523587e-18,-1.764750498535746>, 1 }
    sphere {  m*<-4.386139727839583,8.164965809277259,-2.1938016958174353>, 1}
    sphere { m*<-4.386139727839583,-8.164965809277259,-2.193801695817439>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.44255231990211974,-2.905001727934062e-18,4.052568499960156>, <0.3878763662811349,-5.6313500753238994e-18,1.0530649992515728>, 0.5 }
    cylinder { m*<7.917923627050284,2.4264882103523587e-18,-1.764750498535746>, <0.3878763662811349,-5.6313500753238994e-18,1.0530649992515728>, 0.5}
    cylinder { m*<-4.386139727839583,8.164965809277259,-2.1938016958174353>, <0.3878763662811349,-5.6313500753238994e-18,1.0530649992515728>, 0.5 }
    cylinder {  m*<-4.386139727839583,-8.164965809277259,-2.193801695817439>, <0.3878763662811349,-5.6313500753238994e-18,1.0530649992515728>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    